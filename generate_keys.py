import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["Yash Patel", "Tarun Mukesh", "M Ghous", "M Faraaz", " Naveen", "K Bhatt"]
usernames = ["yash", "tarun", "ghous", "faraaz", "naveen", "bhatt"]
passwords = ["XXXX", "XXXX", "XXXX", "XXXX", "XXXX", "XXXX"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)