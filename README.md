# PDF QA Chatbot

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

An AI-powered system that answers questions about your PDF documents using Google Gemini and Pinecone.

![Demo Screenshot](/screenshot.png) <!-- Add screenshot later -->

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
