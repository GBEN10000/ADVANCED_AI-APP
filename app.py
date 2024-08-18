import asyncio
import aiohttp
import openai
import cv2
import face_recognition
import os
import pickle
import pyttsx3
import speech_recognition as sr
import threading

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage

# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)  # Adjust speed
engine.setProperty('volume', 1)  # Max volume

# Select a voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Choose a different voice if available

# Define the speak function with a more polished tone
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Set up OpenAI API key
openai.api_key = "your_api_key"

async def search_openai_api(user_query):
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {openai.api_key}'},
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': user_query}]
                }
            )
            data = await response.json()
            return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def load_known_faces():
    known_faces = []
    if os.path.exists("known_faces.pickle"):
        with open("known_faces.pickle", "rb") as f:
            known_faces = pickle.load(f)
    else:
        face_dir = "known_faces"
        if os.path.exists(face_dir):
            for file_name in os.listdir(face_dir):
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    image = face_recognition.load_image_file(os.path.join(face_dir, file_name))
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        known_faces.append((os.path.splitext(file_name)[0], encoding[0]))
        with open("known_faces.pickle", "wb") as f:
            pickle.dump(known_faces, f)
    return known_faces

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        speak("Sorry, I did not catch that. Could you please repeat?")
        return None
    except sr.RequestError:
        speak("Sorry, I'm unable to connect to the speech recognition service.")
        return None

def save_known_faces(known_faces):
    with open("known_faces.pickle", "wb") as f:
        pickle.dump(known_faces, f)

def recognize_speech():
    text = record_audio()
    if text:
        return text
    else:
        return None

class MyApp(App):
    def build(self):
        layout = FloatLayout()

        gif_image = AsyncImage(source='askme.jpg', size_hint=(1, 1), pos_hint={'top': 1})
        layout.add_widget(gif_image)

        self.query_label = Label(
            text="Query: ",
            size_hint=(0.8, 0.05),
            pos_hint={'x': 0.1, 'y': 0.7},
            color=(1, 1, 1, 1),
            font_size='18sp'
        )
        layout.add_widget(self.query_label)

        self.response_label = Label(
            text="Response: ",
            size_hint=(0.8, 0.05),
            pos_hint={'x': 0.1, 'y': 0.65},
            color=(1, 1, 1, 1),
            font_size='18sp'
        )
        layout.add_widget(self.response_label)

        self.button = Button(
            text='Start AI',
            size_hint=(0.5, 0.1),
            pos_hint={'center_x': 0.5, 'y': 0},
            background_normal='',
            background_color=(139/255, 0, 0, 1),
            color=(0, 1, 0, 1),
            font_size='20sp',
            bold=True,
            border=(10, 20, 20, 10),
            background_down='down_button.jpg'
        )
        self.button.bind(on_press=self.start_ai)
        layout.add_widget(self.button)

        Window.clearcolor = (0.2, 0.2, 0.2, 1)

        return layout

    def start_ai(self, instance):
        threading.Thread(target=self.main_process).start()

    def main_process(self):
        video_capture = cv2.VideoCapture(0)
        known_faces = load_known_faces()

        def async_process_frame(loop):
            async def process_frame():
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    rgb_frame = frame[:, :, ::-1]
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    for face_encoding in face_encodings:
                        name = "Unknown"
                        match_found = False

                        for known_face in known_faces:
                            known_encoding = known_face[1]
                            if len(known_encoding) == 128:
                                distance = face_recognition.face_distance([known_encoding], face_encoding)
                                if distance[0] < 0.6:
                                    name = known_face[0]
                                    match_found = True
                                    break

                        if not match_found:
                            speak("I didn't see you before. Could you please tell me, what is your name?")
                            user_name = recognize_speech()
                            if user_name:
                                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                                known_faces.append((user_name, face_encoding))
                                save_known_faces(known_faces)
                                filename = f"{user_name}.jpg"
                                cv2.imwrite(filename, frame)
                                speak(f"{user_name}, it's an honor to meet you.")
                        if match_found:
                            speak(f"Do you have any doubts, {name}?")
                            user_query = recognize_speech()

                            if user_query:
                                self.query_label.text = f"Query: {user_query}"

                                if user_query in ['who are you', 'what is your name', 'who created you', 'tell me about yourself']:
                                    speak("I am Tsuki, an AI assistant designed to help you with various tasks. I was created by Gunturu Chintu. I can assist with answering questions, searching Wikipedia, and more.")
                                else:
                                    speak(f"Do you mean {user_query}?")
                                    user_response = recognize_speech()

                                    if user_response:
                                        user_response = user_response.lower()
                                        self.response_label.text = f"Response: {user_response}"

                                        if user_response in ['no', 'no i dont', 'i dont']:
                                            speak("Sorry, could you please repeat?")
                                            user_query = recognize_speech()
                                            self.query_label.text = f"Query: {user_query}"
                                        elif user_response in ['yes', 'yes i do', 'i do']:
                                            try:
                                                response = await search_openai_api(user_query)
                                                speak(response)
                                                self.response_label.text = f"Response: {response}"
                                                print(f"Response: {response}")
                                            except Exception as e:
                                                speak(f"Error occurred: {str(e)}")
                                                print(f"Error occurred: {str(e)}")
                                    else:
                                        print("No valid response received.")

                            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            break

                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                video_capture.release()
                cv2.destroyAllWindows()

            # Run the async function in the new event loop
            asyncio.set_event_loop(loop)
            loop.run_until_complete(process_frame())

        # Create a new event loop for the thread
        loop = asyncio.new_event_loop()
        threading.Thread(target=async_process_frame, args=(loop,)).start()

    def show_popup(self, title, message):
        popup_layout = BoxLayout(orientation='vertical', padding=10)
        popup_label = Label(text=message)
        close_button = Button(text='Close', size_hint=(1, 0.2))
        popup_layout.add_widget(popup_label)
        popup_layout.add_widget(close_button)
        popup = Popup(title=title, content=popup_layout, size_hint=(0.6, 0.4))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

if __name__ == '__main__':
    MyApp().run()
