import os

import pandas as pd

path = os.path.abspath("data/example_masse.xlsx")
data = pd.read_excel(path)

text = []
for idx in data.index:
    text.append(f"'{data["Masse"][idx]}':{data["Pesi"][idx]},")# ! this can be changed

with open("packages/masse.py", "w") as file:
    file.write(f"masse = {{{"".join(text)}}}")
