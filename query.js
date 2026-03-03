import * as dotenv from 'dotenv';
dotenv.config();
import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

// 1. Initialize with API Key
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY.trim() });
const History = [];

async function transformQuery(question) {
    // We use a temporary array so we don't mess up the main History yet
    const tempContents = [...History, { role: 'user', parts: [{ text: question }] }];

    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: tempContents,
        config: {
            systemInstruction: `You are a query rewriting expert. Rephrase the user's latest question into a standalone version that includes all necessary context from the history. Output ONLY the rephrased question.`,
        },
    });

    // V2 SDK Access Pattern
    return response.candidates[0].content.parts[0].text;
}

async function chatting(question) {
    try {
        const queries = await transformQuery(question);
        
        // 2. Fix Dimensions to match your 3072 Pinecone Index
        const embeddings = new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GEMINI_API_KEY,
            modelName: 'embedding-001', // Native 3072-dim model
        });

        const queryVector = await embeddings.embedQuery(queries);

        const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY.trim() });
        const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME.trim());

        const searchResults = await pineconeIndex.query({
            topK: 5,
            vector: queryVector,
            includeMetadata: true,
        });

        const context = searchResults.matches
            .map(match => match.metadata?.text || "")
            .join("\n\n---\n\n");

        // Prepare History for the final answer
        History.push({ role: 'user', parts: [{ text: queries }] });

        const response = await ai.models.generateContent({
            model: "gemini-2.0-flash",
            contents: History,
            config: {
                systemInstruction: `You are a DSA Expert. Answer the question based ONLY on this context:
                
                ${context}
                
                If not found, say "I could not find the answer in the provided document."`,
            },
        });

        const answer = response.candidates[0].content.parts[0].text;

        // Save model response to History
        History.push({ role: 'model', parts: [{ text: answer }] });

        console.log("\nBOT:", answer, "\n");

    } catch (error) {
        console.error("Error in chatting:", error.message);
    }
}

async function main() {
    console.log("--- DSA Chatbot (Type 'exit' to stop) ---");
    while (true) {
        const userProblem = readlineSync.question("Ask me anything --> ");
        if (userProblem.toLowerCase() === 'exit') break;
        if (!userProblem.trim()) continue;

        await chatting(userProblem);
    }
}

main();