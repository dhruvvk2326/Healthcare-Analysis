import pandas as pd
import spacy as sp


df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv",usecols=['Disease','Fatigue','Difficulty Breathing'])

titles =  [_ for _ in df['Disease']]
#print(titles)

def find_text(text):
    return "B" in text

g= (title for title in titles if find_text(title))
result = []
for i in range(len(titles)):
    try:
        result.append(next(g))
    except StopIteration:
        break

print(result)
nlp= sp.load ("en_core_web_sm")

x=[a for a in nlp("My name is Mehul.")]
doc = nlp("My name is Mehul.")
print(x)

