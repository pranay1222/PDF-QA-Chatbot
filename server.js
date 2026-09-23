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

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

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

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const sessionStore = new Map();

// Helper function to create a delay and prevent rate limiting (429 errors)
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

app.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const pdfPath = req.file.path;
    console.log(`Processing PDF: ${pdfPath}`);
    
    const sessionId = Date.now().toString();
    const namespace = `pdf-${sessionId}`;
    
    const pdfLoader = new PDFLoader(pdfPath);
    const rawDocs = await pdfLoader.load();
    console.log(`Loaded ${rawDocs.length} document pages`);
    
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    
    const validChunks = chunkedDocs.map(doc => {
      doc.pageContent = doc.pageContent.replace(/\u0000/g, '').trim();
      return doc;
    }).filter(doc => doc.pageContent.length > 0);
    
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      modelName: 'gemini-embedding-2',
      outputDimensionality: 768
    });
    
    console.log(`Generating vectors for ${validChunks.length} chunks safely...`);
    
    const validVectors = [];
    const validDocs = [];
    
    for (let i = 0; i < validChunks.length; i++) {
      try {
        const text = validChunks[i].pageContent;
        
        // Wait 2 seconds between requests to respect the free tier rate limit
        await delay(2000);
        
        const vector = await embeddings.embedQuery(text);
        
        if (vector && vector.length === 768) {
          validVectors.push(vector);
          validDocs.push(validChunks[i]);
        } else {
          console.log(`WARNING: Chunk ${i} rejected. Vector length: ${vector?.length || 0}`);
        }
      } catch (err) {
        console.error(`API Error on chunk ${i}:`, err.message);
      }
    }
    
    if (validVectors.length === 0) {
      throw new Error("Failed to generate any valid 768-dimension embeddings.");
    }
    
    const vectorStore = new PineconeStore(embeddings, {
      pineconeIndex,
      namespace: namespace
    });
    
    await vectorStore.addVectors(validVectors, validDocs);
    console.log(`Successfully indexed ${validVectors.length} chunks in namespace: ${namespace}`);
    
    sessionStore.set(sessionId, {
      namespace,
      history: []
    });
    
    res.json({ message: 'PDF processed successfully', sessionId });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'Failed to process PDF: ' + error.message });
  }
});

app.post('/ask', async (req, res) => {
  try {
    const { question, sessionId } = req.body;
    if (!question || !sessionId) {
      return res.status(400).json({ error: 'Missing question or session ID' });
    }
    
    const session = sessionStore.get(sessionId);
    if (!session) {
      return res.status(400).json({ error: 'Invalid session ID' });
    }
    
    const { namespace, history } = session;
    const transformedQuery = await transformQuery(question, history);
    
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      modelName: 'gemini-embedding-2',
      outputDimensionality: 768
    });
    
    const queryVector = await embeddings.embedQuery(transformedQuery);
    
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: pinecone.Index(process.env.PINECONE_INDEX_NAME),
      namespace: namespace
    });
    
    const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 10);
    const relevantResults = results.filter(([_, score]) => score >= 0.5);
    
    let context = '';
    if (relevantResults.length > 0) {
      context = relevantResults
        .map(([doc]) => doc.pageContent)
        .filter(text => text.trim().length > 0)
        .join("\n\n---\n\n");
    }
    
    let answer;
    if (context) {
      answer = await generateAnswer(question, context);
    } else {
      answer = "I couldn't find the answer in the document.";
    }
    
    history.push({ role: 'user', parts: [{ text: question }] });
    history.push({ role: 'model', parts: [{ text: answer }] });
    
    res.json({ answer });
  } catch (error) {
    console.error('Ask error:', error);
    res.status(500).json({ error: 'Failed to process question: ' + error.message });
  }
});

async function transformQuery(question, history) {
  try {
    if (history.length === 0) return question;
    
    const tempHistory = [...history];
    tempHistory.push({ role: 'user', parts: [{ text: question }] });
    
    const model = genAI.getGenerativeModel({ 
      model: "gemini-3.6-flash",
      systemInstruction: `Rephrase the user's question into a standalone question based on the conversation history. Only output the rewritten question.`
    });
    
    const result = await model.generateContent({ contents: tempHistory });
    const response = await result.response;
    return response.text();
  } catch (error) {
    return question;
  }
}

async function generateAnswer(question, context) {
  try {
    const prompt = `Context: ${context}\n\nQuestion: ${question}\n\nAnswer the question using ONLY the context above. If the answer cannot be found in the context, say "I couldn't find the answer in the document."`;
    
    const model = genAI.getGenerativeModel({ 
      model: "gemini-3.6-flash",
      generationConfig: { temperature: 0.3, maxOutputTokens: 500 }
    });
    
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const answer = response.text();
    
    if (answer.toLowerCase().includes("couldn't find") || answer.toLowerCase().includes("don't know")) {
      return "I couldn't find the answer in the document.";
    }
    return answer;
  } catch (error) {
    return "I couldn't find the answer in the document.";
  }
}

function cleanupUploads() {
  const uploadDir = path.join(__dirname, 'uploads');
  if (fs.existsSync(uploadDir)) {
    fs.rmSync(uploadDir, { recursive: true, force: true });
    console.log('Cleaned up uploads directory');
  }
}

app.use((err, req, res, next) => {
  console.error('Global error handler caught:', err);
  res.status(500).json({ error: err.message || 'An unexpected server error occurred' });
});

app.listen(port, () => {
  cleanupUploads();
  console.log(`Server running on http://localhost:${port}`);
});