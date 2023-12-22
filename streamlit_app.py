import json
import streamlit as st

from openai import OpenAI


def main():
    st.set_page_config(page_title="Chat With A Person")
    st.header("Let's Chat !", divider="gray")

    f = open('character.json')
    data = json.load(f)
    f.close()

    # Initialize the message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "client" not in st.session_state:
        st.session_state.client = None

    # Side Bar Settings
    settings_tab, personality_tab = st.sidebar.tabs(["Settings", "Personality"])

    # Settings Tab
    with settings_tab:
        st.title("Settings")

        api_key = st.text_input("Api Key", type="password")

        # Set horizontal columns
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            model = st.selectbox(
                "Model",
                ("gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct")
            )
        with col2_2:
            lang = st.selectbox(
                "Language",
                ("id", "en"),
                placeholder="Indonesia"
            )

        st.session_state.client = OpenAI(api_key=api_key)

    # Personality Tab
    with (personality_tab):
        st.title("Personality")

        if st.toggle("Template", value=True):
            template = st.selectbox("Character",
                                    ("Lila", "Jovan", "Vincent", "Siti"),
                                    placeholder="Lila")

            personalities = data[lang]['character'][template]

        else:
            # Set the information needed
            name = st.text_input("Name", placeholder="Vincent")

            col1_1, col1_2 = st.columns(2)
            with col1_1:
                gender = st.selectbox(
                    "Gender",
                    ("Laki - Laki", "Perempuan"),
                    placeholder="Laki - Laki"
                )
            with col1_2:
                age = st.text_input("Age", placeholder="20")

            city = st.text_input("City", placeholder="Jakarta")
            personality = st.text_area("Personality", placeholder="Kind, Talktive, ...", max_chars=200)
            behaviour = st.text_area("Behaviour", placeholder="Usually said ooh at the end of the text", max_chars=200)
            like = st.text_area("Like", placeholder="Food, Golf, ...", max_chars=200)
            dislike = st.text_area("Dislike", placeholder="Bad People, Spicy Food", max_chars=200)

            personalities = (
                str(data[lang]['template']).replace("{name}", name).
                replace("{gender}", gender).replace("{age}", age).
                replace("{city}", city).replace("{behaviour}", behaviour).
                replace("{personality}", personality).replace("{like}", like).replace("{dislike}", dislike)
            )

        if st.button("Chat", type="primary", ):

            st.session_state.messages = [
                {"role": "system", "content": f"{personalities}"}
            ]

    # View and update chat
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # React to the chat
    if prompt := st.chat_input("Say something"):
        # Add the user text to the history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Get the response
            for response in st.session_state.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                    top_p=1,
                    frequency_penalty=0.45,
                    presence_penalty=0.15
            ):
                # Stream response
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")

            # Replace the response with the complete one
            message_placeholder.markdown(full_response)

        # Add the bot respond to the history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == '__main__':
    main()
