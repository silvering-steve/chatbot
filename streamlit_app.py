import datetime
import json
import requests

import streamlit as st

from openai import OpenAI


def main():
    st.set_page_config(page_title="Chat With A Person")
    st.header("Let's Chat !", divider="gray")

    if "client" not in st.session_state:
        st.session_state.client = None

    if "data" not in st.session_state:
        st.session_state.data = {
            "person": {},
            "family": {},
            "behaviour": {},
            "respond": {}
        }

    if "personality" not in st.session_state:
        st.session_state.personality = None

    with open("question.json", 'r') as file:
        question_file = json.load(file)
        questions = [question["question"] for question in question_file["questions"]]

        file.close()

    chat_tab, data_tab, personality_tab, example_tab = st.tabs(["Chat", "Data", "Personality", "Example"])

    with data_tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Name")
        with col2:
            age = st.text_input("Age")
        with col3:
            date = st.date_input("Birthdate", min_value=datetime.datetime.today() - datetime.timedelta(days=60*365))

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            birthplace = st.text_input("Birthplace")
        with col5:
            address = st.text_input("Address")
        with col6:
            city = st.text_input("Live City")
        with col7:
            religion = st.text_input("Religion")

        col8, col9 = st.columns(2)
        with col8:
            job = st.text_input("Job")
        with col9:
            job_desc = st.text_area("Job Desc")

        st.divider()

        col10, col11, col12, col13 = st.columns(4)
        with col10:
            father = st.text_input("Father")
        with col11:
            mother = st.text_input("Mother")
        with col12:
            sister = st.text_input("Sister")
        with col13:
            brother = st.text_input("Brother")

        st.divider()

        likes = st.text_area("Likes", max_chars=100)
        dislikes = st.text_area("Dislikes", max_chars=100)
        personalities = st.text_area("Personality", max_chars=100)
        behaviour = st.text_area("Talking Behaviours", max_chars=100)

        st.divider()

        dlike_respond = st.text_area("Usual respond with something that doesn't like", max_chars=100)
        like_respond = st.text_area("Usual respond with something that like", max_chars=100)
        dknow_respond = st.text_area("Usual respond with something that doesn't know", max_chars=100)

        st.session_state.data["person"]["name"] = name
        st.session_state.data["person"]["age"] = age
        st.session_state.data["person"]["date"] = str(date)
        st.session_state.data["person"]["birthplace"] = birthplace
        st.session_state.data["person"]["address"] = address
        st.session_state.data["person"]["city"] = city
        st.session_state.data["person"]["religion"] = religion
        st.session_state.data["person"]["job"] = job
        st.session_state.data["person"]["job_desc"] = job_desc

        st.session_state.data["family"]["father"] = father
        st.session_state.data["family"]["mother"] = mother
        st.session_state.data["family"]["brother"] = brother
        st.session_state.data["family"]["sister"] = sister

        st.session_state.data["behaviour"]["likes"] = likes
        st.session_state.data["behaviour"]["dislikes"] = dislikes
        st.session_state.data["behaviour"]["personalities"] = personalities
        st.session_state.data["behaviour"]["behaviour"] = behaviour

        st.session_state.data["respond"]["d_like"] = dlike_respond
        st.session_state.data["respond"]["like"] = like_respond
        st.session_state.data["respond"]["d_know"] = dknow_respond

        json_data_person = json.dumps(st.session_state.data, indent=2)
        st.download_button(
            label="Download Person",
            file_name="data_person.json",
            mime="application/json",
            data=json_data_person,
        )

    with personality_tab:
        choices = ["Disagree", "Slightly disagree", "Neutral", "Slightly agree", "Agree"]
        personality = {}

        for idx, question in enumerate(questions, start=1):
            personality[f"{idx}"] = choices.index(st.select_slider(
                question,
                ["Disagree", "Slightly disagree", "Neutral", "Slightly agree", "Agree"],
                value="Neutral"
            )) + 1

        if st.button("Predict", type="primary"):
            headers = {
                "content-type": "application/json",
                "X-RapidAPI-Key": "73393e472bmsh711d065d4b7aa16p17ca0bjsncbce18697459",
                "X-RapidAPI-Host": "big-five-personality-test.p.rapidapi.com"
            }

            payload = {
                "answers": personality
            }

            response = requests.post(
                url="https://big-five-personality-test.p.rapidapi.com/submit",
                json=payload,
                headers=headers
            )

            st.session_state.personality = st.session_state.client.chat.completions.create(
                            model="gpt-3.5-turbo-1106",
                            messages=[
                                {"role": "system", "content": "You are a personality helper bot that can easily summarize people personalities based on the data that will be given using simple words and without using 'based on the data' word."},
                                {"role": "user", "content": f"{response.json()} summarize it and retell it like you talking to somebody else and no number within 100 words"}
                            ],
                            stream=False,
                            top_p=1,
                            frequency_penalty=0.45,
                            presence_penalty=0.15
                    ).choices[0].message.content

            st.markdown(st.session_state.personality)

    with example_tab:
        e_casual = {}
        for i in range(3):
            e_casual[i] = st.text_area(f"Example chat when talking casually {i + 1}")

        e_likes = {}
        for i in range(3):
            e_likes[i] = st.text_area(f"Example chat when talking something he likes {i + 1}")

        e_dlikes = {}
        for i in range(3):
            e_dlikes[i] = st.text_area(f"Example chat when talking something he doesn't likes {i + 1}")

        e_dknow = {}
        for i in range(3):
            e_dknow[i] = st.text_area(f"Example chat when talking something he don't know {i + 1}")

        if st.button("Done"):
            dataset = []

            system_casual = f"You are {st.session_state.data['person']['name']}, a {st.session_state.data['person']['age']}-years-old {st.session_state.data['person']['job']}, {st.session_state.personality}"
            system_likes = f"You are {st.session_state.data['person']['name']}, a {st.session_state.data['person']['age']}-years-old {st.session_state.data['person']['job']}, {st.session_state.personality} You likes {st.session_state.data['behaviour']['likes']}, and when talking about something you likes usually you {st.session_state.data['respond']['like']}"
            system_dlikes = f"You are {st.session_state.data['person']['name']}, a {st.session_state.data['person']['age']}-years-old {st.session_state.data['person']['job']}, {st.session_state.personality} You dislikes {st.session_state.data['behaviour']['dislikes']}, and when talking about something you dislikes usually you {st.session_state.data['respond']['d_like']}"
            system_dknow = f"You are {st.session_state.data['person']['name']}, a {st.session_state.data['person']['age']}-years-old {st.session_state.data['person']['job']}, {st.session_state.personality} You usually {st.session_state.data['behaviour']['behaviour']}, When talking about something you dont know usually you {st.session_state.data['respond']['d_know']}"

            for chat in e_casual.values():
                messages = {"messages": [{
                    "role": "systems", "content": system_casual
                }]}

                text = chat.splitlines()

                messages['messages'].append({"role": "user", "content":text[0].replace("U:", "")})
                messages['messages'].append({"role": "assistant", "content": text[1].replace("A:", "")})
                messages['messages'].append({"role": "user", "content": text[2].replace("U:", "")})
                messages['messages'].append({"role": "assistant", "content": text[3].replace("A:", "")})

                dataset.append(messages)

            for chat in e_likes.values():
                messages = {"messages": [{
                    "role": "systems", "content": system_likes
                }]}

                text = chat.splitlines()

                messages['messages'].append({"role": "user", "content":text[0].replace("U:", "")})
                messages['messages'].append({"role": "assistant", "content": text[1].replace("A:", "")})
                messages['messages'].append({"role": "user", "content": text[2].replace("U:", "")})
                messages['messages'].append({"role": "assistant", "content": text[3].replace("A:", "")})

                dataset.append(messages)

            for chat in e_dlikes.values():
                messages = {"messages": [{
                    "role": "systems", "content": system_dlikes
                }]}

                text = chat.splitlines()

                messages['messages'].append({"role": "user", "content":text[0].replace("U:", "")})
                messages['messages'].append({"role": "assistant", "content": text[1].replace("A:", "")})
                messages['messages'].append({"role": "user", "content": text[2].replace("U:", "")})
                messages['messages'].append({"role": "assistant", "content": text[3].replace("A:", "")})

                dataset.append(messages)

            for chat in e_dknow.values():
                messages = {"messages": [{
                    "role": "systems", "content": system_dknow
                }]}

                text = chat.splitlines()

                messages['messages'].append({"role": "user", "content":text[0].replace("U:", "")})
                messages['messages'].append({"role": "assistant", "content": text[1].replace("A:", "")})
                messages['messages'].append({"role": "user", "content": text[2].replace("U:", "")})
                messages['messages'].append({"role": "assistant", "content": text[3].replace("A:", "")})

                dataset.append(messages)

            st.write(dataset)

            json_data = json.dumps(dataset, indent=2)
            st.download_button(
                label="Download JSON",
                file_name="data.json",
                mime="application/json",
                data=json_data,
            )

    with st.sidebar:
        st.title("Settings")

        api_key = st.text_input("Api Key", type="password")

        if st.button("Submit"):
            st.session_state.client = OpenAI(api_key=api_key)

    # f = open('character.json')
    # data = json.load(f)
    # f.close()
    #
    # Initialize the message history
    # if "messages" not in st.session_state:
    #     st.session_state.messages = []
    #

    #
    # Data
    # name = st.text_input("Name", placeholder="Steven Adi Santoso")
    #
    # col1, col2 = st.columns(2)
    # with col1:
    #     age = st.text_input("Age", placeholder=22)
    #
    # with col2:
    #     birth = st.date_input("Birth")
    #
    # # Relation
    # relations = st.text_input("Relation with user")
    # family = st.text_area("Family relations")

    # # Side Bar Settings
    # settings_tab, personality_tab = st.sidebar.tabs(["Settings", "Personality"])
    #
    # # Settings Tab
    # with settings_tab:
    #     st.title("Settings")
    #
    #     api_key = st.text_input("Api Key", type="password")
    #
    #     # Set horizontal columns
    #     col2_1, col2_2 = st.columns(2)
    #     with col2_1:
    #         model = st.selectbox(
    #             "Model",
    #             ("gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct")
    #         )
    #     with col2_2:
    #         lang = st.selectbox(
    #             "Language",
    #             ("id", "en"),
    #             placeholder="Indonesia"
    #         )
    #
    #     st.session_state.client = OpenAI(api_key=api_key)
    #
    # # Personality Tab
    # with (personality_tab):
    #     st.title("Personality")
    #
    #     if st.toggle("Template", value=True):
    #         template = st.selectbox("Character",
    #                                 ("Lila", "Jovan", "Vincent", "Siti"),
    #                                 placeholder="Lila")
    #
    #         personalities = data[lang]['character'][template]
    #
    #     else:
    #         # Set the information needed
    #         name = st.text_input("Name", placeholder="Vincent")
    #
    #         col1_1, col1_2 = st.columns(2)
    #         with col1_1:
    #             gender = st.selectbox(
    #                 "Gender",
    #                 ("Laki - Laki", "Perempuan"),
    #                 placeholder="Laki - Laki"
    #             )
    #         with col1_2:
    #             age = st.text_input("Age", placeholder="20")
    #
    #         city = st.text_input("City", placeholder="Jakarta")
    #         personality = st.text_area("Personality", placeholder="Kind, Talktive, ...", max_chars=200)
    #         behaviour = st.text_area("Behaviour", placeholder="Usually said ooh at the end of the text", max_chars=200)
    #         like = st.text_area("Like", placeholder="Food, Golf, ...", max_chars=200)
    #         dislike = st.text_area("Dislike", placeholder="Bad People, Spicy Food", max_chars=200)
    #
    #         personalities = (
    #             str(data[lang]['template']).replace("{name}", name).
    #             replace("{gender}", gender).replace("{age}", age).
    #             replace("{city}", city).replace("{behaviour}", behaviour).
    #             replace("{personality}", personality).replace("{like}", like).replace("{dislike}", dislike)
    #         )
    #
    #     if st.button("Chat", type="primary", ):
    #
    #         st.session_state.messages = [
    #             {"role": "system", "content": f"{personalities}"}
    #         ]
    #
    # # View and update chat
    # for message in st.session_state.messages:
    #     if message["role"] != "system":
    #         with st.chat_message(message["role"]):
    #             st.markdown(message["content"])
    #
    # # React to the chat
    # if prompt := st.chat_input("Say something"):
    #     # Add the user text to the history
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #
    #     # Display user message in container
    #     with st.chat_message("user"):
    #         st.markdown(prompt)
    #
    #     with st.chat_message("assistant"):
    #         message_placeholder = st.empty()
    #         full_response = ""
    #
    #         # Get the response
    #         for response in st.session_state.client.chat.completions.create(
    #                 model=model,
    #                 messages=[
    #                     {"role": m["role"], "content": m["content"]}
    #                     for m in st.session_state.messages
    #                 ],
    #                 stream=True,
    #                 top_p=1,
    #                 frequency_penalty=0.45,
    #                 presence_penalty=0.15
    #         ):
    #             # Stream response
    #             full_response += (response.choices[0].delta.content or "")
    #             message_placeholder.markdown(full_response + "▌")
    #
    #         # Replace the response with the complete one
    #         message_placeholder.markdown(full_response)
    #
    #     # Add the bot respond to the history
    #     st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == '__main__':
    main()
