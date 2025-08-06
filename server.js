require('dotenv').config();
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { GoogleGenerativeAIEmbeddings } = require('@langchain/google-genai');
const { PDFLoader } = require('@langchain/community/document_loaders/fs/pdf');
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const { Pinecone } = require('@pinecone-database/pinecone');
const { PineconeStore } = require('@langchain/pinecone');

const app = express();
const port = process.env.PORT || 3000;

// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'pdf-' + uniqueSuffix + '.pdf');
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 50 * 1024 * 1024 }
});

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});

// Initialize Google GenAI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Store session information
const sessionStore = new Map();

// PDF Upload Endpoint
app.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const pdfPath = req.file.path;
    console.log(`Processing PDF: ${pdfPath}`);
    
    // Generate unique session ID and namespace
    const sessionId = Date.now().toString();
    const namespace = `pdf-${sessionId}`;
    
    const pdfLoader = new PDFLoader(pdfPath);
    const rawDocs = await pdfLoader.load();
    console.log(`Loaded ${rawDocs.length} document pages`);
    console.log(`First page text: ${rawDocs[0]?.pageContent?.substring(0, 100)}...`);
    
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log(`Split into ${chunkedDocs.length} chunks`);
    
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });
    
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    
    // Store in Pinecone with unique namespace
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 5,
      namespace: namespace
    });
    
    console.log(`Documents indexed in namespace: ${namespace}`);
    
    // Store session info
    sessionStore.set(sessionId, {
      namespace,
      history: []
    });
    
    // Get index stats for debugging
    const stats = await pineconeIndex.describeIndexStats();
    console.log(`Index stats: ${JSON.stringify(stats)}`);
    
    res.json({ 
      message: 'PDF processed successfully', 
      sessionId 
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'Failed to process PDF: ' + error.message });
  }
});

// Question Answering Endpoint
app.post('/ask', async (req, res) => {
  try {
    const { question, sessionId } = req.body;
    if (!question || !sessionId) {
      return res.status(400).json({ error: 'Missing question or session ID' });
    }
    
    console.log(`Processing question for session ${sessionId}: ${question}`);
    
    // Get session info
    const session = sessionStore.get(sessionId);
    if (!session) {
      return res.status(400).json({ error: 'Invalid session ID' });
    }
    
    const { namespace, history } = session;
    
    // Transform query
    const transformedQuery = await transformQuery(question, history);
    console.log(`Transformed query: ${transformedQuery}`);
    
    // Get embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });
    
    const queryVector = await embeddings.embedQuery(transformedQuery);
    console.log(`Query vector length: ${queryVector.length}`);
    
    // Create vector store with namespace
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: pinecone.Index(process.env.PINECONE_INDEX_NAME),
      namespace: namespace
    });
    
    // Perform similarity search with scores
    const results = await vectorStore.similaritySearchVectorWithScore(
      queryVector,
      10
    );
    
    console.log(`Found ${results.length} matches`);
    
    // Filter by similarity score
    const similarityThreshold = 0.5;
    const relevantResults = results.filter(
      ([_, score]) => score >= similarityThreshold
    );
    
    console.log(`Relevant matches: ${relevantResults.length}/${results.length}`);
    
    let context = '';
    if (relevantResults.length > 0) {
      context = relevantResults
        .map(([doc]) => doc.pageContent)
        .filter(text => text.trim().length > 0)
        .join("\n\n---\n\n");
        
      console.log(`Context length: ${context.length} characters`);
      console.log(`Context sample: ${context.substring(0, 200)}...`);
    } else {
      console.log('No relevant matches found');
    }
    
    // Generate answer
    let answer;
    if (context) {
      answer = await generateAnswer(question, context);
    } else {
      answer = "I couldn't find the answer in the document.";
    }
    
    // Update history
    history.push({ role: 'user', parts: [{ text: question }] });
    history.push({ role: 'model', parts: [{ text: answer }] });
    
    console.log(`Generated answer: ${answer.substring(0, 100)}...`);
    res.json({ answer });
  } catch (error) {
    console.error('Ask error:', error);
    res.status(500).json({ error: 'Failed to process question: ' + error.message });
  }
});

// Helper function: Transform query
async function transformQuery(question, history) {
  try {
    // If no history, just return the original question
    if (history.length === 0) {
      return question;
    }
    
    const tempHistory = [...history];
    tempHistory.push({ role: 'user', parts: [{ text: question }] });
    
    const model = genAI.getGenerativeModel({ 
      model: "gemini-1.5-flash",
      systemInstruction: `Rephrase the user's question into a standalone question based on the conversation history. 
      Only output the rewritten question.`
    });
    
    const result = await model.generateContent({ contents: tempHistory });
    const response = await result.response;
    return response.text();
  } catch (error) {
    console.error('Transform query error:', error);
    return question;
  }
}

// Generate answer with enhanced context handling
async function generateAnswer(question, context) {
  try {
    // Create a more effective prompt
    const prompt = `Context: ${context}\n\nQuestion: ${question}\n\nAnswer the question using ONLY the context above. If the answer cannot be found in the context, say "I couldn't find the answer in the document."`;
    
    const model = genAI.getGenerativeModel({ 
      model: "gemini-1.5-flash",
      generationConfig: {
        temperature: 0.3,
        maxOutputTokens: 500
      }
    });
    
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const answer = response.text();
    
    // Simple verification
    if (answer.toLowerCase().includes("couldn't find") || 
        answer.toLowerCase().includes("don't know")) {
      return "I couldn't find the answer in the document.";
    }
    
    return answer;
  } catch (error) {
    console.error('Generate answer error:', error);
    return "I couldn't find the answer in the document.";
  }
}

// Cleanup uploaded files on server start
function cleanupUploads() {
  const uploadDir = path.join(__dirname, 'uploads');
  if (fs.existsSync(uploadDir)) {
    fs.rmSync(uploadDir, { recursive: true, force: true });
    console.log('Cleaned up uploads directory');
  }
}

// Start server
app.listen(port, () => {
  cleanupUploads();
  console.log(`Server running on http://localhost:${port}`);
});