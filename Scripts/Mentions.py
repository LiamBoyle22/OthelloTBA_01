import pandas as pd
import glob
import os
import re 
import matplotlib.pyplot as plt

folder = r"C:\Users\liamb\Source\Othello_01\Data\cleaned"

characters = characters = ["Othello", "Desdemona", "Iago", "Cassio", "Emilia", "Roderigo", "Brabantio", "Bianca", "Lodovico", "Gratiano", "Montano"]

rows = []
for path in glob.glob(os.path.join(folder, "*in_character.txt")):
    with open(path, encoding = "utf-8", errors = "replace") as f:
        text = f.read()
    row = {"file": os.path.basename(path)}
    for name in characters:
        row[name] = len(re.findall(r"\b" + name + r"\b", text, flags = re.IGNORECASE))
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("character_mentions_by_file.csv", index=False)

totals = df[characters].sum().sort_values(ascending=False)
print("\nTotal mentions:")
print(totals)

plt.figure(figsize=(10, 6))
plt.bar(totals.index, totals.values)
plt.title("How often each character is mentioned")
plt.xlabel("Character")
plt.ylabel("Number of mentions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("character_mentions.png")
plt.close()

print("\nSaved character_mentions_by_file.csv and character_mentions.png")