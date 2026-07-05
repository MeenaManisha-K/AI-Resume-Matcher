import streamlit as st
import PyPDF2 as pdf
import google.generativeai as genai
import chromadb  # 🚀 UPGRADE: Import Vector Database

def extract_text_from_pdf(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        return text
    except Exception as e:
        return ""

# 🚀 UPGRADE: Helper function to break the text into smaller chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Page Setup
st.set_page_config(page_title="AI Resume Matcher with RAG", page_icon="🤖")
st.title("🤖 Advanced RAG-Powered Gemini AI Resume Matcher")
st.write("This app uses ChromaDB Vector Embeddings and Gemini 2.5 Flash to perform targeted semantic analysis.")

# Secure API Key Input Layout
st.sidebar.subheader("🔑 API Authentication")
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Any Resume")
    uploaded_file = st.file_uploader("Upload resume file (PDF)...", type=["pdf"])

with col2:
    st.subheader("2. Paste Any Job Details")
    job_desc = st.text_area("Paste the requirements or description here...", height=200)

if st.button("Run Real AI Analysis"):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar first!")
    elif uploaded_file is None or job_desc.strip() == "":
        st.warning("Please upload a resume PDF and paste a job description!")
    else:
        with st.spinner("Initializing RAG Pipeline and Vectorizing Text..."):
            try:
                # 1. Configure the live API key dynamically
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                # 2. Extract the raw text from the resume
                resume_text = extract_text_from_pdf(uploaded_file)
                
                # 🚀 UPGRADE 3. IMPLEMENTING THE RAG PIPELINE
                # Step A: Split the raw text into clean text chunks
                text_chunks = chunk_text(resume_text, chunk_size=400)
                
                # Step B: Initialize an ephemeral (in-memory) ChromaDB Client
                chroma_client = chromadb.Client()
                
                # Create a temporary vector collection (clears out on every refresh)
                # We use a default built-in embedding function provided by ChromaDB
                collection = chroma_client.get_or_create_collection(name="resume_analysis")
                
                # Create unique string IDs for each chunk
                chunk_ids = [f"id_{i}" for i in range(len(text_chunks))]
                
                # Add our resume chunks directly into the Vector Database
                collection.add(
                    documents=text_chunks,
                    ids=chunk_ids
                )
                
                # Step C: Search the database! Pull the chunks matching the job requirements
                search_results = collection.query(
                    query_texts=[job_desc],
                    n_results=min(3, len(text_chunks)) # Extract top matching sections
                )
                
                # Combine retrieved contexts into a single string block
                retrieved_context = "\n\n".join(search_results['documents'][0])
                
                # 4. Create the automated AI prompt instruction using retrieved context
                prompt = f"""
                You are an expert ATS (Applicant Tracking System) recruiter and talent acquisition analyst.
                Analyze the following targeted segments from the candidate's resume against the job description using context retrieved from a vector database.
                Do not use generic pre-written answers. Evaluate the specific candidate's text against the specific job text.
                
                Provide your analysis structured exactly like this:
                
                ## 📊 Live AI Recruiter Report (RAG Optimized)
                
                ### Target Domain Categorization
                [State the industry/role type found in the job description]
                
                ### 🎯 Overall Match Rating
                [Give a realistic match percentage out of 100% based on their skills vs the job requirements]
                
                ### 💡 Strengths Identified (Skills Found in Resume)
                [List specific skills, projects, or keywords from this resume that match this job description perfectly]
                
                ### 🔍 Critical Gaps & Missing Keywords
                [List tools, skills, or industry practices explicitly mentioned in the job description that this resume completely lacks]
                
                ### 🎯 Tailored Interview Preparation Track
                1. [Write a highly technical interview question targeting an actual project listed in the resume]
                2. [Write an interview question asking how they would bridge one of the missing gaps identified above]
                3. [Write a situational technical scenario combining their resume background with the job requirements]
                
                ---
                Retrieved Context from Candidate's Vector Profile:
                {retrieved_context}
                
                Target Job Description Content:
                {job_desc}
                """
                
                # 5. Generate the real response from the model
                response = model.generate_content(prompt)
                
                st.success("Analysis Complete via RAG Architecture!")
                st.markdown("---")
                st.markdown(response.text)
                
                # Clean up collection context to reset state
                chroma_client.delete_collection(name="resume_analysis")
                
            except Exception as e:
                st.error(f"Error Processing Request: {str(e)}")