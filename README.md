# AI-Powered-JD-Resume-Analyzer
AI-Powered JD &amp; Resume Analyzer is an intelligent tool that analyzes job descriptions and resumes using NLP and machine learning to generate ATS scores, personalized interview questions, assess compatibility scores, and provide actionable insights for job seekers and recruiters.

💡 Project Idea

Built an AI-based application to analyze resumes against job descriptions, generate role-specific interview questions, and provide intelligent feedback on skill gaps, match scores, and sentiment-based answer quality.

📌 Project Overview
Created a tool that uses NLP, machine learning, and ranking algorithms to:
Analyze job descriptions
Score resumes against the JD
Simulate interview questions (based on job role)
Suggest resume improvements
Provide feedback based on simulated answers

🎯 Problem It Solves
Job seekers struggle to tailor resumes and prepare for interviews effectively. This tool gives data-driven suggestions and mock interview simulations.

🧱 What I've Built
🔹 1. JD-to-Resume Match Engine
Extracts keywords and intent from a JD (NLP using spaCy, transformers)
Analyzes a user's resume PDF and compute a match score
Suggests missing skills or keyword gaps

🔹 2. AI Interview Generator
Generates relevant interview questions based on role + JD using LLM-based prompt engineering or rule-based logic
Allows user to input sample answers (text-based)
Provides sentiment + confidence feedback using models like TextBlob or BERT sentiment

🔹 3. Feedback & Dashboard
Score: resume match %, answer quality, missing skills
Recommendations for improvement
Streamlit dashboard or Flask web interface

🔧 Tech Stack
Python, NLP (spaCy, NLTK, BERT/Transformers)
PDF parsing (PyMuPDF, PDFMiner)
Streamlit or Flask for dashboard
scikit-learn for any ML scoring etc.
