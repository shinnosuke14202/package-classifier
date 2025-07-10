from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

data = pd.read_csv("./data/filtered_dataset.csv")

# Extract features (X) and labels (y)
X = data['appname']
y = data['categorygame'].apply(lambda x: 1 if x.strip().lower() == 'game' else 0)  # convert to binary (1 = game)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline: TF-IDF vectorizer + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
    ('clf', LogisticRegression())
])

# Train
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

print("Test Results:\n", classification_report(y_test, y_pred))

# Try it on new data
examples = [
    "com.infinitygames.hex",                 # likely game
    "com.quicknotes.notepad",               # not game
    "com.speedracer.extreme3d",             # game
    "com.taskbuddy.scheduler",              # not game
    "com.dragonking.battlearena",           # game
    "com.weatherwise.forecastpro",          # not game
    "com.minijump.frenzy",                  # game
    "com.dailytodo.listmaker",              # not game
    "com.warofclans.empiredefense",         # game
    "com.meditate.relaxnow",                # not game
    "com.crashzombies.apocalypse",          # game
    "com.calculatorplus.tools",             # not game
    "com.sandboxcraft.buildnexplore",       # game
    "com.passwordmanager.lockvault",        # not game
    "com.piratelegends.caribbeantreasure",  # game
    "com.ebookreader.libpremium",           # not game
    "com.fps.shootingmissionelite",         # game
    "com.videoplayer.ultra",                # not game
    "com.wizardduel.magicwars",             # game
    "com.scanit.documentscanner",           # not game
    "com.gladiatorclash.arena",             # game
    "com.financely.budgettracker",          # not game
    "com.racingfever.hightorque",           # game
    "com.notesync.cloudeditor",             # not game
    "com.zombiebattle.towerdefense",        # game
    "com.healthyhabits.watertracker",       # not game
    "com.jetfighters.acecombat",            # game
    "com.languagelearner.spanishguru",      # not game
    "com.galacticfleet.spacewars",          # game
    "com.sleepcycle.alarmpro",              # not game
    "com.match3jewels.classicfun",          # game
    "com.cookingmaster.chefchallenge",      # game
    "com.myjournal.dailyentries",           # not game
    "com.shadowninja.stealthattack",        # game
    "com.pdfscanner.camplus",               # not game
    "com.battleknights.rpgquest",           # game
    "com.worktimer.focusboost",             # not game
    "com.snakevsblock.challenge",           # game
    "com.routeplanner.mapsnavigator",       # not game
    "com.stickmanfight.mortalcombat",       # game
    "com.airhorn.pranktools",               # not game
    "com.zombiehunter.laststand",           # game
    "com.barcode.readerlite",               # not game
    "com.craftblock.exploration3d",         # game
    "com.focusreader.newsfeed",             # not game
    "com.gunshooter.sniperlegends",         # game
    "com.remindme.taskalert",               # not game
    "com.dragonsaga.fantasyquest",          # game
    "com.musicstream.hiphopbeats",          # not game
    "com.mathquiz.braintrainer",            # game
    "com.towerclash.battledefense",         # game
]

predictions = model.predict(examples)

for pkg, label in zip(examples, predictions):
    print(f"{pkg} -> {'Game' if label == 1 else 'Not Game'}")
