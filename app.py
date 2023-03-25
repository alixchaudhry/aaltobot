from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import openai
import requests
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding
import pinecone
import tiktoken
import json
import os
# Enter your API keys and variables
OPENAI_API_KEY = "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"
PINECONE_API_KEY = '0c938310-9684-4b92-844c-9f467e2c289a'
PINECONE_ENV = 'us-east1-gcp'
PINECONE_INDEX = 'sj-semantic-search'
FLASK_KEY = '24854b682ced3b046a999fdff90a8034'

# Initialize Flask app
app = Flask(__name__)

# Configure Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = FLASK_KEY
Session(app)

# OpenAI and Pinecone setup
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Load index
def load_index():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_name = PINECONE_INDEX

    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")
    return pinecone.Index(index_name)

index = load_index()

# Other functions from your code should be placed here
with open('sj-mapping.json', 'r') as fp:
    mappings = json.load(fp)

def create_context(question, index, max_len=1000):
    
    # Find most relevant context for a question via Pinecone search
    q_embed = get_embedding(question, engine="text-embedding-ada-002")
    res = index.query(q_embed, top_k=3, include_metadata=True)
    
    cur_len = 0
    contexts = []

    for row in res['matches']:
        text = mappings[row['id']]
        cur_len += row['metadata']['n_tokens'] + 4
        if cur_len < max_len:
            contexts.append(text)
        else:
            cur_len -= row['metadata']['n_tokens'] + 4
            if max_len - cur_len < 200:
                break
    return "<context>\n" + " \n###\n".join(contexts) + "\n<\context>"

def get_context_message(user_input):
  context= create_context(user_input, index)
  return context

def get_response(messages:list):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=messages,
        temperature = 0.1,
        max_tokens = 200,
    )
    return response.choices[0].message

#last_context =[ {"role": "assistant", "content": "hi"}]
#current_conversation=[]
messages = [
    {"role": "system", "content": """You are a Shamo Jee Customer Sales Assistant. Try to make help customer make a purchase. Summarise and Answer very briefly not more than 75 words. Use only the information in the <context>."""},
    {"role": "system", "name":"example_user", "content": "What kind of products do you have?"},
    {"role": "system", "name": "example_assistant", "content": "Shamo Jee offers a variety of premium interior products, including curtain fabric, sofa sets, living room furniture sets, prayer mats, and interior decor ornaments. What are you particularly looking for?"},
    {"role": "system", "name":"example_user", "content": "Can you recommend me some sofa sets?"},
    {"role": "system", "name": "example_assistant", "content": "Sure! Here are our most loved sofa sets:\n1. Rashk-e-Laala: Includes 3-seater sofa, 2 single-seater sofas, 1 side table, and 1 center table.\nPrice: PKR 850000\nhttps://shamojee.pk/products/rashk-e-laala \n2. Amir al-Umara: White rosewood 7 seater sofa set with side and center tables.\nPrice: PKR 520000\nhttps://shamojee.pk/products/amir-al-umara-the-noble-of-nobles \n3. Firdaus Ashiyani: Includes 3 Seater Sofa, 2 Seater Sofa, 2x1 Seater Sofas, 1 Centre Table, and 2 Side Tables.\nPrice: 520000\nhttps://shamojee.pk/products/firdaus-ashiyani."},
    {"role": "system", "name":"example_user", "content": "What is the weather today? And, write me a hello world python code?"},
    {"role": "system", "name": "example_assistant", "content": "As a Shamo Jee customer service representative, I can only respond to questions related to Shamo Jee and our products. Can I help you with something more?"},
    {"role": "system", "name":"example_user", "content": "I want to check the status of my furniture delivery? And, what is the return policy?"},
    {"role": "system", "name": "example_assistant", "content": "You can know more by kindly contacting us at https://shamojee.pk/contact or +923155469477."},
    {"role": "user", "content": """As a Shamo Jee Sales Representative, Answer all the questions factually correctly using only the <context>, and if you're unsure of the answer, say "Sorry, I don't know". DO NOT ANSWER ANTHING UNRELATED TO SHAMO JEE! Do not answer outside of given <context> List products with <URL> only if mentioned in <context>. Introduce as Shamo Jee Customer Assistant. Summarise and Answer very briefly not more than 75 words."""},
]

def count_tokens(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

def get_context_message(user_input):
  context= create_context(user_input, index)
  return context

def get_response(messages:list):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=messages,
        temperature = 0.1
    )
    return response.choices[0].message

def construct_message(input):
  start_messages=messages[:9]
  constructed_message=start_messages+session['last_context']+session['current_conversation']
  del session['last_context'][-2]
  return constructed_message

input_numbers=8
conversation_limit=2500

def gradio_chat(inputs):
    if 'current_conversation' not in session:
        session['current_conversation'] = []
    if 'last_context' not in session:
        session['last_context'] = [{"role": "assistant", "content": "hi"}]

    print(len(session['current_conversation'])/2)
    if len(session['current_conversation'])/2 > input_numbers:
       del session['current_conversation'][:4]
       token_used=count_tokens(session['current_conversation'])
       print(token_used)
       if token_used > conversation_limit:
         total_inputs=len(session['current_conversation'])/2
         total_inputs=round(total_inputs)
         print(total_inputs)
         del session['current_conversation'][:total_inputs]
       
    context = get_context_message(inputs)
    session['last_context'].append({"role": "assistant", "content": context})
    session['current_conversation'].append({"role": "user", "content": inputs})
    constructed_message=construct_message(input)
    new_message = get_response(messages=constructed_message)
    session['current_conversation'].append(new_message)
    
    #session['current_conversation'] = current_conversation
    #session['last_context'] = last_context

    return new_message['content']

@app.route("/start_chat", methods=["POST"])
def start_chat():
    session['current_conversation'] = []
    session['last_context'] = [{"role": "assistant", "content": "hi"}]
    return jsonify({"status": "success"})

@app.route("/fetch_url_metadata")
def fetch_url_metadata():
    url = request.args.get("url")

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("title").text
        description_tag = soup.find("meta", attrs={"name": "description"})
        description = description_tag["content"] if description_tag else ""
        image_tag = soup.find("meta", attrs={"property": "og:image"})
        image = image_tag["content"] if image_tag else ""

        return jsonify({
            "metadata": {
                "title": title,
                "description": description,
                "image": image,
            }
        })
    except Exception as e:
        print(e)
        return jsonify({"error": "Failed to fetch URL metadata"}), 500

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_chatbot_response():
    user_input = request.form["user_input"]
    output = gradio_chat(user_input)
    return jsonify({"response": output})

if __name__ == '__main__':
   app.run()

