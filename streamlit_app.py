import json
import os
import shutil
import subprocess
from typing import Dict, List

from elevenlabs import set_api_key, Voice, VoiceSettings, generate
from moviepy.video import VideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment

import streamlit as st

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from src.gradio_demo import SadTalker


class Person:
    def __init__(
            self,
            name: str,
            age: str,
            occupation: Dict[str, str],
            likes: str,
            dislikes: str,
            behaviours: str,
            personalities: str,
            respond: List[str],
            birthplace: str = None,
            birthdate: str = None,
            address: str = None,
            religion: str = None
    ):
        self.name = name
        self.age = age
        self.occupation = occupation
        self.likes = likes
        self.dislikes = dislikes
        self.behaviours = behaviours
        self.personalities = personalities
        self.respond = respond
        self.birthplace = birthplace
        self.birthdate = birthdate
        self.address = address
        self.religion = religion

        self.template = f"""
            Your name is {self.name}:
            - Age: {self.age}.
            - Birthplace: {self.birthplace}
            - Birthday: {self.birthdate}
            - Occupation: {self.occupation["title"]}.
            - Occupation Description: {self.occupation["desc"]}.
            - Likes: {self.likes}.
            - Dislikes: {self.dislikes}.
            - Personalities: {self.personalities}.
            - Behaviours when talking casually: You usually reply with {self.behaviours} when talking.
            - Behaviours when talking something you like: You usually reply with {self.respond[0]} behaviours.
            - Behaviours when talking something you didn't like: You usually reply with {self.respond[1]} behaviours.
            - Behaviours when talking something you don't know: You usually reply with {self.respond[2]} behaviours.

        """


class Audio:
    def __init__(
            self,
            voice_id: str,
            stability: float,
            similarity: float,
            style: float
    ):
        self.voice_id = voice_id
        self.stability = stability
        self.similarity = similarity
        self.style = style

    def __call__(self, text: str, *args, **kwargs):
        return generate(
            text=text,
            voice=Voice(
                voice_id=self.voice_id,
                settings=VoiceSettings(
                    stability=self.stability,
                    similarity_boost=self.similarity,
                    style=self.style,
                    use_speaker_boost=True
                )
            )
        )


class Video:
    def __init__(self, name, enhance, crop, pose, exp, still, blink):
        self.model = SadTalker(lazy_load=True)

        self.name = name
        self.enhance = enhance
        self.crop = crop
        self.pose = pose
        self.exp = exp
        self.still = still
        self.blink = blink

        os.makedirs("data", exist_ok=True)

    def __call__(self, *args, **kwargs):
        return self.model.test(
            source_image=f"data/{self.name}/face.jpg",
            driven_audio=f"data/{self.name}/audio.wav",
            use_enhancer=self.enhance,
            preprocess=self.crop,
            pose_style=self.pose,
            facerender='facevid2vid',
            exp_scale=self.exp,
            use_ref_video=False,
            ref_video=None,
            ref_info=None,
            still_mode=self.still,
            use_idle_mode=False,
            length_of_audio=0, use_blink=self.blink,
            result_dir=f'./results/{self.name}'
        )


def main():
    st.set_page_config(page_title="Chat With A Person")
    st.header("Let's Chat !", divider="gray")

    if "chain" not in st.session_state:
        st.session_state.chain = None

    if "memory" not in st.session_state:
        st.session_state.memory = []

    if "audio" not in st.session_state:
        st.session_state.audio = None

    if "video" not in st.session_state:
        st.session_state.video = None

    if "person" not in st.session_state:
        st.session_state.person = None

    with st.sidebar:
        st.title("Settings")

        api_tab, chat_tab, eleven_tab, face_tab = st.tabs(["API", "Chat", "Eleven Labs", "Face"])
        with api_tab:
            eleven_api = st.text_input("Eleven Labs API Key")
            chat_api = st.text_input("Chat GPT API Key")

            if st.button("Set API"):
                set_api_key(eleven_api)
        with chat_tab:
            model_id = st.text_input("Model ID")
            temp = st.slider("Temperature", max_value=2.0, min_value=0.0, value=0.8)
            file = st.file_uploader("Person Data", type=["json"])

            if file:
                person_data = json.loads(file.read())

                with st.expander("Person"):
                    st.write(person_data)

            if st.button("Set Model"):
                st.session_state.person = Person(
                    name=person_data["name"],
                    age=person_data["age"],
                    occupation={"title": person_data["job"], "desc": person_data["job_desc"]},
                    likes=person_data["likes"],
                    dislikes=person_data["dislikes"],
                    behaviours=person_data["behaviour"],
                    personalities=person_data["personalities"],
                    birthplace=person_data["birthplace"],
                    birthdate=person_data["birthdate"],
                    address=person_data["address"],
                    respond=list((person_data["like_respond"], person_data["dlike_respond"],
                                  person_data["dknow_respond"]))
                )

                prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(
                            content=st.session_state.person.template
                        ),  # The persistent system prompt
                        MessagesPlaceholder(
                            variable_name="chat_history"
                        ),  # Where the memory will be stored.
                        HumanMessagePromptTemplate.from_template(
                            "{input}"
                        ),  # Where the human input will injected
                    ]
                )
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                model = ChatOpenAI(
                    model=model_id,
                    openai_api_key=chat_api,
                    temperature=temp
                )
                st.session_state.chain = LLMChain(
                    llm=model,
                    prompt=prompt,
                    verbose=True,
                    memory=memory
                )
                st.session_state.memory = []
        with eleven_tab:
            voice_id = st.text_input("Voice ID")

            cola, colb, colc = st.columns(3)
            with cola:
                stability = st.slider("Stability", max_value=1.0, value=0.7)
            with colb:
                similarity = st.slider("Similarity", max_value=1.0, value=0.6)
            with colc:
                style = st.slider("Style", max_value=1.0, value=0.2)

            if st.button("Set Voices"):
                st.session_state.audio = Audio(
                    voice_id,
                    stability,
                    similarity,
                    style
                )
        with face_tab:
            if face := st.file_uploader("Face Image", type=["jpg"]):
                os.makedirs(f"data/{st.session_state.person.name}", exist_ok=True)

                with open(f"data/{st.session_state.person.name}/face.jpg", "wb") as image:
                    image.write(face.read())

            cold, cole, colf = st.columns(3)
            with cold:
                enhance = st.checkbox("Enhance Face")
            with cole:
                blink = st.checkbox("Enable Blink")
            with colf:
                still = st.checkbox("Enable Still Mode")

            colg, colh = st.columns(2)
            with colg:
                exp = st.slider("Exp Scale", min_value=0.0, max_value=2.0, value=1.0)
            with colh:
                pose = st.slider("Pose Style", min_value=0, max_value=40)

            crop = st.selectbox("Preprocess", ["crop", "full"])

            if st.button("Set Video"):
                with st.spinner("Loading..."):
                    st.session_state.video = Video(
                        name=st.session_state.person.name,
                        enhance=enhance,
                        blink=blink,
                        still=still,
                        exp=exp,
                        pose=pose,
                        crop=crop,
                    )

                    if os.path.exists(f"results/{st.session_state.person.name}"):
                        shutil.rmtree(f"results/{st.session_state.person.name}", )

    for message in st.session_state.memory:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Say something"):
        # Add the user text to the history
        st.session_state.memory.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            output = st.session_state.chain.predict(input=prompt)
            st.session_state.memory.append({"role": "assistant", "content": output})

            audio = st.session_state.audio(output)
            with open(f"data/{st.session_state.person.name}/audio.wav", "wb") as audio_file:
                audio_file.write(audio)
            audio_file.close()

            video = st.session_state.video()
            ffmpeg_command = [
                'ffmpeg',
                '-y',  # Overwrite existing output file
                '-i', video,
                '-c:v', 'libx265',
                '-crf', '23',  # Adjust the CRF value as needed
                '-c:a', 'aac',
                '-b:a', '128k',  # Adjust the audio bitrate as needed
                f"{video}_264.mp4"
            ]

            # Run FFmpeg command
            subprocess.run(ffmpeg_command)

            st.write(output)
            st.audio(audio)
            st.video(f"{video}_264.mp4")


if __name__ == '__main__':
    main()
