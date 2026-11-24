import re
from pathlib import Path
import matplotlib.pyplot as plt

folder = Path(r"C:/Users/liamb/Source/Othello_01/Data/cleaned")

characters = ["Othello", "Desdemona", "Iago", "Cassio", "Emilia", "Roderigo",
              "Brabantio", "Bianca", "Lodovico", "Gratiano", "Montano"]
themes = ["black", "jealous", "handkerchief", "wife", "honest"]

def count(word, text):
    return len(re.findall(rf"\b{word}\b", text, re.IGNORECASE))

performed = " ".join(f.read_text(errors="ignore") for f in folder.glob("*in_character.txt"))
for c in characters:
    print(c, count(c, performed))

files = sorted(folder.glob("*.txt"), key=lambda f: int(re.search(r"\d+", f.stem).group()))
for c in characters:
    plt.plot([count(c, f.read_text(errors="ignore")) for f in files], label=c)

plt.xlabel("Session")
plt.ylabel("Mentions in discussion")
plt.legend()
plt.savefig("fig_theme_trends.png", dpi=300)