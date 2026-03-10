import * as dotenv from 'dotenv';
dotenv.config();
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import pdf from 'pdf-parse-fork';

async function debugEmbed(){
  const PDF_PATH = './dsa.pdf';
  const pdfLoader = new PDFLoader(PDF_PATH, { pdfParse: pdf });
  const rawDocs = await pdfLoader.load();
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
  const validDocs = chunkedDocs.filter(doc => doc.pageContent && doc.pageContent.trim().length > 0);
  
  const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'gemini-embedding-2-preview',
  });

  const BATCH_SIZE = 10;
  for (let i = 0; i < validDocs.length; i += BATCH_SIZE) {
      const batch = validDocs.slice(i, i + BATCH_SIZE).map(d => d.pageContent);
      console.log(`Embedding batch ${i} to ${i + batch.length - 1}...`);
      try {
          const res = await embeddings.embedDocuments(batch);
          for (let j = 0; j < res.length; j++) {
              if (res[j].length !== 3072) {
                  console.error(`ERROR: Chunk ${i+j} returned dimension ${res[j].length}! Content snippet: ${batch[j].substring(0, 50)}`);
              }
          }
      } catch (err) {
          console.error(`ERROR on batch ${i}:`, err.message);
      }
      await new Promise(r => setTimeout(r, 2000)); // sleep to avoid massive 429
  }
}
debugEmbed();
