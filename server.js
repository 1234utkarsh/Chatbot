import * as dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json());
app.use(express.static(__dirname)); // Serve index.html from project root

// Shared chat history (per server session)
const History = [];

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY.trim() });

async function transformQuery(question) {
    const tempContents = [...History, { role: 'user', parts: [{ text: question }] }];
    const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents: tempContents,
        config: {
            systemInstruction: `You are a query rewriting expert. Rephrase the user's latest question into a standalone version that includes all necessary context from the history. Output ONLY the rephrased question.`,
        },
    });
    return response.candidates[0].content.parts[0].text;
}

app.post('/api/chat', async (req, res) => {
    const { question } = req.body;
    if (!question || !question.trim()) {
        return res.status(400).json({ error: 'Question is required.' });
    }

    try {
        const standaloneQuery = await transformQuery(question);

        const embeddings = new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GEMINI_API_KEY,
            modelName: 'text-embedding-004',
            outputDimensionality: 3072,
        });

        const queryVector = await embeddings.embedQuery(standaloneQuery);

        const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY.trim() });
        const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME.trim());

        const searchResults = await pineconeIndex.query({
            topK: 5,
            vector: queryVector,
            includeMetadata: true,
        });

        const context = searchResults.matches
            .map(match => match.metadata?.text || '')
            .join('\n\n---\n\n');

        History.push({ role: 'user', parts: [{ text: standaloneQuery }] });

        const response = await ai.models.generateContent({
            model: 'gemini-2.0-flash',
            contents: History,
            config: {
                systemInstruction: `You are a DSA Expert. Answer the question based ONLY on this context:\n\n${context}\n\nIf the answer is not in the context, say "I could not find the answer in the provided document."`,
            },
        });

        const answer = response.candidates[0].content.parts[0].text;
        History.push({ role: 'model', parts: [{ text: answer }] });

        res.json({ answer });
    } catch (error) {
        console.error('Error:', error.message);
        res.status(500).json({ error: 'Something went wrong. Please try again.' });
    }
});

// Clear conversation history
app.post('/api/clear', (req, res) => {
    History.length = 0;
    res.json({ message: 'History cleared.' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`DSA Chatbot server running at http://localhost:${PORT}`);
});
