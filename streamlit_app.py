import streamlit as st

from openai import OpenAI


def main():
    st.set_page_config(page_title="Chat With A Person")
    st.header("Let's Chat !", divider="gray")

    # Initialize the message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "client" not in st.session_state:
        st.session_state.client = None

    # Side Bar Settings
    st.sidebar.title("Settings")

    # Set the information needed
    name = st.sidebar.text_input("Name", placeholder="Vincent")
    personalities = st.sidebar.text_area("Personalities", placeholder="You are a helpful assistant bot")

    # Set horizontal columns
    col1, col2 = st.sidebar.columns(2)
    with col1:
        model = st.selectbox(
            "Model",
            ("gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct")
        )
    with col2:
        lang = st.selectbox(
            "Language",
            ("Indonesia", "English")
        )

    api_key = st.sidebar.text_input("Api Key", type="password")

    if st.sidebar.button("Chat", type="primary",):
        match lang:
            case "Indonesia":
                intro = f"Namamu adalah {name}"
            case "English":
                intro = f"Your name is {name}"

        st.session_state.messages = [
            {"role": "system", "content": f"{intro}. {personalities}"}
        ]

        st.session_state.client = OpenAI(api_key=api_key)

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
