# PDF QA Chatbot
https://pdf-qa-chatbot-x020.onrender.com/

An AI-powered system that answers questions about your PDF documents using Google Gemini and Pinecone.

screenshots:

<img width="1250" height="901" alt="Screenshot (308)" src="https://github.com/user-attachments/assets/87a5b53f-ae45-472a-8559-3b2decc87159" />
<img width="1280" height="916" alt="Screenshot (309)" src="https://github.com/user-attachments/assets/a295a145-c0e6-4866-b31f-f38c6790bfb3" />
<img width="1229" height="906" alt="Screenshot (310)" src="https://github.com/user-attachments/assets/d546a28d-ccb5-4a78-bcbf-0fa231927a63" />


## Features
- PDF upload with drag & drop
- Natural language questioning
- Context-aware answers
- Conversation history
- Mobile-responsive UI

## Tech Stack
- **Frontend**: HTML/CSS/JavaScript
- **Backend**: Node.js/Express
- **AI**: Google Gemini
- **Vector DB**: Pinecone

## Installation
```bash
git clone https://github.com/pranay1222/pdf-qa-chatbot.git
cd pdf-qa-chatbot
npm install
```

## Configuration
Create `.env` file:
```env
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
GEMINI_API_KEY=your_gemini_key
PORT=3000
```

## Usage
```bash
node server.js
```
Visit `http://localhost:3000`
