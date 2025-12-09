import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel
import math
import re
import os
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import openai
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from google import generativeai as genai
from dotenv import load_dotenv

# Load model keys from env
load_dotenv()  # loads .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_eval_model():
    # We use GPT-2 to calculate Perplexity (how well it predicts text)
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

# --- METRIC 1: PERPLEXITY CALCULATOR ---
def calculate_perplexity(text, model, tokenizer):
    """
    Perplexity = exp(Cross Entropy Loss).
    Lower is better. A PPL of 10 means the model is as confused as if it had to pick from 10 words.
    $$ Perplexity= e^{Cross Entropy Loss} $$
    Cross-entropy tells us: how surprised the model is by the correct next token.
    """
    #tokenize the text into pytorch tensors pt
    encodings = tokenizer(text, return_tensors="pt")
    
    # max_length for GPT-2 is 1024. We clamp it for the demo. Truncate to GPT-2 max length
    input_ids = encodings.input_ids[:, :1024]
    
    #Since we’re evaluating, not training, we don’t need gradient computation
    with torch.no_grad(): 
        """Forward pass with labels
        input_ids: inputs for the model to read (tokenized text)
        labels:  for the model to predict
        You are saying:
        “Using the text, predict the next token for every position — 
        and compare your predictions to this same text.”
        But important:
        HuggingFace automatically shifts the labels by one position inside the model.
        setting labels as input id tells the model:
        “Predict the next token for each position in this sequence.”
        Internally, the model shifts labels by 1 position:
        Input IDs:   [A, B, C, D]
        Labels:      [B, C, D, <eos>]
        So the model learns:
        From A → predict B
        From AB → predict C
        From ABC → predict D
        The model returns a loss automatically when labels are provided.
        """
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    #.item() converts a PyTorch tensor to a Python float.
    ppl = torch.exp(loss).item()
    return ppl

# --- METRIC 2: N-GRAM OVERLAP (ROUGE-1 Proxy) ---
def simple_rouge_score(reference, candidate):
    """
    Calculates the overlap of words (unigrams) between reference and candidate.
    Recall = (Overlapping Words) / (Total Words in Reference)
    Precision = (Overlapping Words) / (Total Words in Candidate)
    F1 = Harmonic Mean
    """
    # Simple tokenization by splitting on whitespace and removing punctuation
    def tokenize(text):
        """
        Converts text to lowercase
        Removes punctuation using regex
        ([^\w\s] removes anything that is not a word character or whitespace)
        Splits text on whitespace into words
        Converts list of words into a set
        → This removes duplicates
        """
        text = re.sub(r'[^\w\s]', '', text.lower())
        return set(text.split())

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    overlap = ref_tokens.intersection(cand_tokens)
    #recall is the ratio of overlapping words to the total number of words in the reference
    #What % of reference is covered
    recall = len(overlap) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    #precision is the ratio of overlapping words to the total number of words in the candidate
    #What % of candidate is covered
    precision = len(overlap) / len(cand_tokens) if len(cand_tokens) > 0 else 0
    #f1 is the harmonic mean of recall and precision
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
        
    return recall, precision, f1

# --- METRIC 2: ROUGE ---
@st.cache_data
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# --- METRIC 3: BERTScore ---
@st.cache_data
def compute_bertscore(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang='en', rescale_with_baseline=True)
    return P[0].item(), R[0].item(), F1[0].item()

# --- METRIC 4: Embedding Similarity ---
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def compute_embedding_similarity(reference, candidate, model):
    ref_emb = model.encode(reference, convert_to_tensor=True)
    cand_emb = model.encode(candidate, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(ref_emb, cand_emb).item()
    return similarity

# --- LLM as a Judge ---
def grade_with_openai(question, answer):
    prompt = f"""
    You are an impartial judge. Rate the quality of the answer on a scale of 1-10.

    Question: {question}
    Student Answer: {answer}

    Score (1-10):
    Reasoning:
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Access content using dot notation (v1.x standard)
        output_text = response.choices[0].message.content
        return output_text

    except Exception as e:
        return f"Error: {e}"

def grade_with_gemini(question, answer):
    prompt = f"""
    You are an impartial judge. Rate the quality of the answer on a scale of 1-10.

    Question: {question}
    Student Answer: {answer}

    Score (1-10):
    Reasoning:
    """
    
    try:
        # Using gemini-1.5-flash (fast & efficient) or gemini-pro
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Gemini Error: {e}"

# --- MAIN APP ---
def app():
    st.title("📊 Lab 5: Evaluation & Metrics")
    st.markdown("""
    Welcome! This lab helps you **evaluate how "smart" a language model is**.  
    We will explore metrics from **mathematical**, **reference-based**, and **AI judgment** perspectives.
    """)

    tab_ppl, tab_rouge, tab_judge = st.tabs([
        "1️⃣ Perplexity (Math)", 
        "2️⃣ Reference Metrics (ROUGE)",
        "3️⃣ LLM-as-a-Judge (AI)"
    ])

    # Load Model
    with st.spinner("Loading Evaluator Model..."):
        tokenizer, model = load_eval_model()
    st.success("Evaluator model loaded successfully!")

    # ==========================
    # TAB 1: PERPLEXITY
    # ==========================
    with tab_ppl:
        st.subheader("The 'Confusion' Score – Perplexity")
        st.markdown("""
        Perplexity measures how **surprised a language model is by a given text**.  
        - Think of it like a **teacher trying to predict the next word**:  
          - **Low perplexity** → the text is fluent and predictable  
          - **High perplexity** → the text is confusing, jumbled, or grammatically incorrect  
        """)
        st.markdown("**Algorithm (Simplified):**")
        st.code("""
1. Tokenize the text into words/tokens.
2. Feed tokens into the language model.
3. Compute cross-entropy loss for predicting each next token.
4. Exponentiate the loss to get perplexity.
5. Lower value = better fluency.
""", language="text")

        col1, col2 = st.columns(2)
        with col1:
            text_a = st.text_area("Text A (Fluent English)", 
                                  "The quick brown fox jumps over the lazy dog.", height=100)
        with col2:
            text_b = st.text_area("Text B (Broken/Gibberish)", 
                                  "Dog lazy the over jumps fox brown quick The.", height=100)
            
        if st.button("Calculate Perplexity"):
            ppl_a = calculate_perplexity(text_a, model, tokenizer)
            ppl_b = calculate_perplexity(text_b, model, tokenizer)
            
            c1, c2 = st.columns(2)
            c1.metric("PPL (Text A)", f"{ppl_a:.2f}", delta="Low Confusion (Good)", delta_color="normal")
            c2.metric("PPL (Text B)", f"{ppl_b:.2f}", delta="High Confusion (Bad)", delta_color="inverse")
            
            st.info("Notice: Text B has the exact same words as Text A, but the **order is scrambled**, causing high perplexity. Analogy: the teacher gets confused because the sentence structure is broken.")

    # ==========================
    # TAB 2: REFERENCE METRICS (ROUGE)
    # ==========================
    with tab_rouge:
        st.subheader("Comparing to a 'Gold Standard' – ROUGE")
        st.markdown("""
        ROUGE measures **overlap between a candidate text and a reference text**.  
        - Commonly used in translation, summarization, and text generation evaluation.  
        - Analogy: imagine comparing a student's essay to the teacher's model answer.
        """)
        
        st.markdown("**How each ROUGE type works:**")
        st.markdown("""
        - **ROUGE-1 (Unigrams):** Measures word-level overlap.  
        *Analogy:* Did the student use the same keywords as the model answer?
        - **ROUGE-2 (Bigrams):** Measures two-word sequence overlap.  
        *Analogy:* Did the student use correct phrases, not just isolated words?
        - **ROUGE-L (Longest Common Subsequence):** Measures sequence similarity.  
        *Analogy:* Did the student maintain the correct order of concepts, even if words differ?
        """)
        
        st.markdown("**Algorithm (Simplified Steps):**")
        st.code("""
    1. Tokenize reference and candidate text into unigrams (ROUGE-1) and bigrams (ROUGE-2).  
    2. Compute the **Longest Common Subsequence** for ROUGE-L.  
    3. Count overlapping tokens or sequences:
    - Recall = Overlap / Total tokens in Reference
    - Precision = Overlap / Total tokens in Candidate
    - F1 = Harmonic mean of Precision and Recall
    4. Higher F1 → candidate closely matches reference text.
    """, language="text")
        
        ref_text = st.text_area("Gold Standard Reference (Human)", "The Eiffel Tower is located in Paris, France.")
        cand_text = st.text_area("Model Output (Candidate)", "The Eiffel Tower stands in the city of Paris.")
        
        if st.button("Calculate ROUGE Scores"):
            # Compute all ROUGE types
            scores = compute_rouge(ref_text, cand_text)
            
            st.success("ROUGE evaluation completed!")
            st.markdown("### ROUGE-1 (Unigram Overlap)")
            st.write(f"Precision: {scores['rouge1'].precision:.2%} | Recall: {scores['rouge1'].recall:.2%} | F1: {scores['rouge1'].fmeasure:.2%}")
            st.markdown("### ROUGE-2 (Bigram Overlap)")
            st.write(f"Precision: {scores['rouge2'].precision:.2%} | Recall: {scores['rouge2'].recall:.2%} | F1: {scores['rouge2'].fmeasure:.2%}")
            st.markdown("### ROUGE-L (Longest Common Subsequence)")
            st.write(f"Precision: {scores['rougeL'].precision:.2%} | Recall: {scores['rougeL'].recall:.2%} | F1: {scores['rougeL'].fmeasure:.2%}")
            
            st.caption("""
            **Notes:**  
            - ROUGE measures **word/sequence overlap**, not meaning.  
            - Paraphrased or reworded answers may score lower even if they are correct.  
            - Analogy: The student explained the concept correctly but used different words or phrases from the textbook.
            """)


    # ==========================
    # TAB 3: LLM-AS-A-JUDGE
    # ==========================
    with tab_judge:
        st.subheader("The Modern Way: AI Grading AI")
        st.markdown("""
        Standard metrics like ROUGE cannot capture nuance.  
        The industry trend: **use a powerful AI (GPT-4 and Gemini) as a judge** to evaluate answers.
        
        Analogy: the AI acts like an expert teacher grading student essays.
        """)
        
        st.markdown("#### How the AI Judge Works (Simplified Algorithm)")
        st.code("""
        1. Input: Question + Candidate Answer
        2. GPT-4 reads both.
        3. Evaluates correctness, clarity, completeness, and fluency.
        4. Outputs:
        - Score (1-10)
        - Explanation/Reasoning
        """, language="text")
                
        st.markdown("#### Try it yourself")
        q = st.text_input("Question", "Explain gravity.")
        a = st.text_area("Student Answer", "Gravity is when things fall down because the earth is heavy.")
        
        st.info("Press 'Grade Answer' to have GPT-4 and Gemini provide a **score** and **reasoning**. Make sure your OpenAI API key is in `.env`.")

        if st.button("Grade Answer"):
            # Create two columns for side-by-side comparison
            col_gpt, col_gemini = st.columns(2)
            
            with st.spinner("Judges are grading..."):
                # Call both models
                result_gpt = grade_with_openai(q, a)
                result_gemini = grade_with_gemini(q, a)
            
            st.success("Grading completed!")
            
            with col_gpt:
                st.subheader("🤖 GPT-4 Verdict")
                st.info(result_gpt)
                
            with col_gemini:
                st.subheader("✨ Gemini Verdict")
                st.info(result_gemini)
                
            st.balloons()
