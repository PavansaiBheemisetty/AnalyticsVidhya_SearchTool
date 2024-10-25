import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

file_path = 'cleaned_data.csv'
df = pd.read_csv(file_path)

model = SentenceTransformer('all-MiniLM-L6-v2')

df['CombinedText'] = df.apply(lambda row: f"Title: {str(row['CourseTitle'])}. Level: {str(row['Level']) if pd.notnull(row['Level']) else 'Not specified'}. Duration: {str(row['Time(Hours)']) if pd.notnull(row['Time(Hours)']) else 'Unknown'} hours. Category: {str(row['Category']) if pd.notnull(row['Category']) else 'Not specified'}. Description: {str(row['Description']) if pd.notnull(row['Description']) else 'No description available'}", axis=1)

embeddings = model.encode(df['CombinedText'].tolist(), show_progress_bar=True)

import faiss
import numpy as np

embedding_dim = embeddings.shape[1]

index = faiss.IndexFlatL2(embedding_dim)

index.add(np.array(embeddings))


def refine_query_with_llm(query):
   
    prompt = f"Given the query '{query}', generate a refined query to find relevant courses on this topic."

   
    gemini_api_url = "https://api.gemini.com/v1/generate-text"  
    headers = {
        "Authorization": f"Bearer AIzaSyBgKa2DzyZ7dkqcnXjG-1-vZhkjcZNmi9k", 
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 50
    }
    response = requests.post(gemini_api_url, headers=headers, json=data)
    response_data = response.json()


    refined_query = response_data.get('text', query).strip()
    return refined_query

def search_courses_with_llm(query, top_k=5):

    refined_query = refine_query_with_llm(query)
    print(f"Refined Query: {refined_query}")  


    query_embedding = model.encode([refined_query])[0]


    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for idx in indices[0]:
        course = {
            'CourseTitle': df.iloc[idx]['CourseTitle'],
            'Description': df.iloc[idx]['Description'],
            'Level': df.iloc[idx]['Level'],
            'Category': df.iloc[idx]['Category'],
            'NumberOfLessons': df.iloc[idx]['NumberOfLessons']
        }
       
        results.append(course)
    return results



import gradio as gr

def gradio_search(query):
    results = search_courses_with_llm(query, top_k=5)
    display_results = [
        f"Title: {res['CourseTitle']}\nDescription: {res['Description']}\nCategory: {res['Category']}\nLevel: {res['Level']}"
        for res in results
    ]
    return "\n\n".join(display_results)

# Create Gradio interface without the Flag button and modify the input label
interface = gr.Interface(fn=gradio_search, inputs=gr.Textbox(label="Search in our free courses"), outputs="text", title="Analytics Vidhya", flagging_mode=None)
interface.launch(debug=True)
