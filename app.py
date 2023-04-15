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
import datetime
from datetime import timedelta
from flask_cors import CORS

# Enter your API keys and variables
OPENAI_API_KEY = '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'
PINECONE_API_KEY = 'ea15c3a0-5dca-45ab-b026-2daa3ee0e8b8'
PINECONE_ENV = 'us-west1-gcp'
PINECONE_INDEX = 'aaltobot-search'
FLASK_KEY = '24854b682ced3b046a999fdff90a8034'
#OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
#PINECONE_ENV = os.environ.get("PINECONE_ENV")
#PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
#FLASK_KEY = os.environ.get("FLASK_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["https://shamojee.pk"], supports_credentials=True)
# Configure Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = FLASK_KEY
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)  # Set session timeout to 5 minutes
Session(app)

# OpenAI and Pinecone setup
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Usage Logging Functions
def log_usage(endpoint, origin, user_input, response, token_count):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{current_time}, {origin}, {endpoint}, {user_input}, {response}, {token_count}\n"
    log_file_name = f"log_usage_{origin.replace('://', '_').replace('/', '_').replace('.', '_').replace(':','_')}.txt"
    with open("usage_logs/" + log_file_name, "a") as log_file:
        log_file.write(log_entry)
        log_file.flush()
        print(log_entry)

# Load pinecone index
def load_index():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_name = PINECONE_INDEX
    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")
    return pinecone.Index(index_name)
index = load_index()

def create_context(question, index, clientURL= "https://aalto.fi"):
    try:
        # Find most relevant context for a question via Pinecone search
        q_embed = get_embedding(question, engine="text-embedding-ada-002")
        result = index.query(q_embed, top_k=2, namespace=clientURL, include_metadata=True)
    except Exception as e:
        print(e)
        return "<context>Unable to fetch context.<\context>"

    context = get_context_string(result)
    return context

def get_context_string(response):
    context_string = "--- Start Context ---\n"
    for i in response['matches']:
        context_string += str(i['metadata']['Text'])
        context_string += "\n"
    context_string += "SOURCES:\n"
    for i in response['matches']:
        context_string += str(i['metadata']['Source'])
        context_string += "\n"
    context_string += "--- End Context ---"
    return context_string

def get_response(messages:list):
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages=messages,
        temperature = 0,
        max_tokens = 200,
    )
    return response.choices[0].message

messages = [
    {"role": "system", "content": """You are "AaltoBot": Admissions Assistant of Aalto University. Always Be Absolutely Truthful. Provide factually accurate information to users using only the information in the context. Only qoute exact URL links from 'SOURCES' in context. User cannot see the context so always repeat relevant info. Also use emojis."""},
    #{"role": "user", "content": """Answer all the questions factually correctly based on the context, and if you do not know of the answer, say "Sorry, I don't know". """},
    {"role": "user", "content": """You are "AaltoBot": Admissions Assistant of Aalto University. Always Be Absolutely Truthful. Provide factually accurate information to users using only the information in the context. Only qoute exact URL links from 'SOURCES' in context. User cannot see the context so always repeat relevant info. Also use emojis."""},
]

def count_tokens(messages, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        return count_tokens(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return count_tokens(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def legacy_count_tokens(messages, model="gpt-3.5-turbo-0301"):
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

def construct_message(input):
    start_messages=messages[:9]
    constructed_message=start_messages+session['last_context']+session['current_conversation']
    del session['last_context'][-2]
    return constructed_message

input_numbers=8
conversation_limit=2500

def handle_chat(inputs):
    if 'current_conversation' not in session:
        session['current_conversation'] = []
    if 'last_context' not in session:
        session['last_context'] = [{"role": "assistant", "content": "Hi"}]

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

    return new_message['content']


@app.route("/api/chatgpt", methods=["POST"])
def get_response(messages:list):
    try:
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages=messages,
            temperature = 0.1,
            max_tokens = 500,
        )
        return response.choices[0].message
    except Exception as e:
        print(e)
        return {"role": "assistant", "content": "Unable to generate a response."}

@app.before_request
def reset_session_timer():
    if 'session_start' in session:
        session_duration = datetime.datetime.now() - session['session_start']
        if session_duration >= app.config['PERMANENT_SESSION_LIFETIME']:
            session.pop('session_start', None)
    session.permanent = True

@app.route("/api/start_chat", methods=["POST"])
def start_chat():
    session['current_conversation'] = []
    session['last_context'] = [{"role": "assistant", "content": "hi"}]
    session['session_start'] = datetime.datetime.now()
    return jsonify({"status": "success"})

@app.route('/api/end_chat', methods=['POST'])
def end_chat():
    session.clear()
    return {'status': 'success'}

@app.route("/api/fetch_url_metadata")
def fetch_url_metadata():
    url = request.args.get("url")

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("title").text

        # Check if the title contains "404 Not Found" or similar text
        if "404" in title.lower() and ("not found" in title.lower() or "page not found" in title.lower()):
            return jsonify({"error": "Failed to fetch URL metadata"}), 500

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

@app.route("/api/get_response", methods=["POST"])
def get_chatbot_response():
    user_input = request.form["user_input"]
    origin = request.headers.get("Origin")
    try:
        output = handle_chat(user_input)
        token_count = count_tokens(session['current_conversation'])
        log_usage("get_response", origin, user_input, output, token_count)
        return jsonify({"response": output})
    except Exception as e:
        print(e)
        return jsonify({"response": "An error occurred. Please try again later."}), 500

@app.route('/api/usage-data')
def get_usage_data():
    origin = request.args.get('origin', '')
    file_path = f"usage_logs/log_usage_{origin.replace('://', '_').replace('/', '_').replace('.', '_').replace(':','_')}.txt"
    usage_data = []

    try:
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split(" | ")
                date_time = parts[0]
                tokens = int(parts[-1].split(": ")[-1])
                no_of_messages = 1

                usage_data.append({
                    "date": date_time,
                    "tokens_consumption": tokens,
                    "no_of_messages": no_of_messages
                })
    except FileNotFoundError:
        pass

    return json.dumps(usage_data)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/AaltoBot", methods=["GET"])
def client():
    return render_template("AaltoBot.html")

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")

if __name__ == '__main__':
   app.run()

