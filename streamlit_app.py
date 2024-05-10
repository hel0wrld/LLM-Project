import streamlit as st
from groq import Groq
from typing import Generator

st.title("LLaMA 3 chatbot")
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.sidebar.write("This chatbot uses LLaMA 3 8-billion parameters model  "
                 "\n\n Try tweaking the below parameters to play with it")
selected_temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, step=0.1)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "temperature" not in st.session_state:
    st.session_state.temperature = 0

# Detect if temperature changes
if selected_temperature != st.session_state.temperature:
    selected_temperature = st.session_state.temperature

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Generator function for streaming responses
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    stream = client.chat.completions.create(
        messages=[
            {
                "role": m['role'],
                "content": m['content']
            }
            for m in st.session_state.messages[-3:]
        ],
        model="llama3-8b-8192",
        temperature=selected_temperature,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True
    )
    with st.chat_message("assistant"):
        response = st.write_stream(generate_chat_responses(stream))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
