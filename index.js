import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import pdf from 'pdf-parse-fork'; // Essential for Node v22

async function indexDocument() {
  try {
    const PDF_PATH = './dsa.pdf';
    const pdfLoader = new PDFLoader(PDF_PATH, { pdfParse: pdf });
    const rawDocs = await pdfLoader.load();
    console.log("PDF loaded successfully.");

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log(`Created ${chunkedDocs.length} chunks.`);

    // Configure Embeddings to match your 3072-dimension index
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY, // Explicitly pass your custom key name
      modelName: "text-embedding-004",    // Use modelName, not model
      outputDimensionality: 3072,         // FORCE 3072 to match your index
    });

    console.log("Configuring Pinecone...");
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY.trim(), // .trim() handles extra spaces in .env
    });
    
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME.trim());

    console.log("Storing data in Pinecone (this may take a minute)...");
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 5,
    });

    console.log("Data stored successfully!");
  } catch (error) {
    console.error("Critical Error:", error);
  }
}

indexDocument();