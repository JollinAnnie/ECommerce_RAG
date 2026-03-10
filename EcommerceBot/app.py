import streamlit as st
import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- 1. INITIAL SETUP ---
st.set_page_config(page_title="Veridian | AI Fashion Assistant", page_icon="🛍️", layout="wide")
my_key = st.secrets["PINECONE_API_KEY","GROQ_API_KEY"]

INDEX_NAME = "clothing-recommendations"

@st.cache_resource
def init_resources():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize Groq client
        groq_client = Groq(api_key=GROQ_API_KEY)
        return index, model, groq_client
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None, None

index, embed_model, groq_client = init_resources()

def extract_price_limit(text):
    match = re.search(r'(?:under|below|less than|max|budget of)\s*\$?\s*(\d+)', text.lower())
    return float(match.group(1)) if match else None

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("Veridian Engine")
    st.markdown("*An AI-powered fashion assistant that uses Retrieval-Augmented Generation (RAG) " \
    "to instantly match your personal style and budget with curated boutique recommendations*.")
    st.divider()
    st.success("⚡ Powered by Groq (Llama 3 8B)")
    st.success("🟢 Pinecone: Connected")

# --- 3. CHAT INTERFACE ---
st.title("Veridian Fashion Assistant")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. INPUT HANDLING ---
prompt_input = st.chat_input("Ask Veridian about a style or budget...")
user_input = st.session_state.get("clicked_prompt") or prompt_input

if user_input:
    if "clicked_prompt" in st.session_state:
        del st.session_state.clicked_prompt

    # A. CONTEXTUAL BOOSTER
    search_query = user_input
    if len(st.session_state.messages) >= 2:
        last_user_msg = st.session_state.messages[-2]["content"]
        search_query = f"{user_input} {user_input} {last_user_msg}" 

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        assistant_slot = st.empty()
        assistant_slot.markdown("*Thinking...*")

        # B. RETRIEVAL
        query_vector = embed_model.encode(search_query).tolist()
        limit = extract_price_limit(user_input)
        filter_dict = {"price": {"$lte": limit}} if limit else None
        
        results = index.query(vector=query_vector, top_k=5, include_metadata=True, filter=filter_dict)

        unique_items = []
        seen_names = set()
        for match in results['matches']:
            m = match['metadata']
            name = m.get('name')
            if name and name not in seen_names:
                unique_items.append(f"ITEM: {name} | PRICE: ${m.get('price')} | BRAND: {m.get('brand')} | DETAILS: {m.get('text')}")
                seen_names.add(name)
        
        context_text = "\n".join(unique_items[:3]) 

        # C. SYSTEM PROMPT
        system_prompt = f"""
        You are Veridian, a professional fashion stylist. 
        
        STRICT RULES:
        1. NO HALLUCINATIONS: Do not invent names, brands, or prices. ONLY use the items in the CATALOG DATA.
        2. NO GIANT TEXT: Do not use hashtags (#). 
        3. NO EMOJIS.
        
        EXAMPLE OF EXACTLY HOW YOU MUST REPLY:
        Hello! Here are some excellent options for you:
        
        1. **Red Silk Dress** | **$45.00**
           * This elegant piece by Old Navy is perfect for any special occasion.
        ---
        2. **Black Leather Shoes** | **$89.99**
           * A classic and comfortable choice for formal events.
        ---
        Let me know if you want to see anything else!

        ACTUAL CATALOG DATA TO USE:
        {context_text if context_text else "None found."}
        """

        # D. GROQ API INFERENCE
        full_response = ""
        try:
            response_stream = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {'role': 'system', 'content': system_prompt}, 
                    {'role': 'user', 'content': user_input}
                ],
                stream=True,
                temperature=0.1,
                max_tokens=1024
            )
            
            # Groq's streaming format extracts content slightly differently
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    assistant_slot.markdown(full_response + "▌")
                    
            assistant_slot.markdown(full_response)
            
        except Exception as e:
            st.error(f"Inference Error: {e}")


    st.session_state.messages.append({"role": "assistant", "content": full_response})

