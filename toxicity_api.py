from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from transformers import pipeline
from pydantic import BaseModel
from typing import Optional, Dict, List
import os

# ============================================================
# 🚀 Initialize FastAPI App
# ============================================================
app = FastAPI(title="Cyberbullying Detection API")

# ============================================================
# 🌍 Enable CORS + Sessions
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key="supersecretkey123")

# ============================================================
# 🌐 Static Files & Templates
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ============================================================
# 🔐 Login System
# ============================================================
USERS = {"admin": "password123", "user": "cyber2026"}  # Demo users

# Global variables for statistics - FIXED: Moved to top-level scope
analysis_count = 0
toxic_count = 0
analysis_history: List[Dict] = []
#####################################
# Example route for history data
# ============================================================
# 🗂️ Analysis History Endpoint (GET /api/history)
# ============================================================
# Example history route
@app.get("/api/history")
def get_history():
    return [
        {
            "text": "You are stupid!",
            "result": "Toxic",
            "confidence": "97%",
            "category": "Harassment",
            "severity": "High"
        },
        {
            "text": "You are amazing!",
            "result": "Safe",
            "confidence": "95%",
            "category": "Compliment",
            "severity": "Low"
        }
    ]
# Example route for adding new users
@app.post("/api/users/add")
async def add_user(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    role = data.get("role")

    # Save to database or print for now
    print(f"New user added: {username} | Role: {role}")
    return {"message": "User added successfully!"}
########################################


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None):
    """Show login page"""
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


@app.post("/login", response_class=HTMLResponse)
async def login_user(
        request: Request,
        username: str = Form(...),
        password: str = Form(...)
):
    """Handle login form"""
    if username in USERS and USERS[username] == password:
        request.session["user"] = username
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid credentials"}
    )


@app.get("/logout")
async def logout_user(request: Request):
    """Log out and redirect"""
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

# ============================================================
# 🏠 Home Page (Public)
# ============================================================
@app.get("/home", response_class=HTMLResponse)
async def home_page(request: Request):
    """Public home page"""
    return templates.TemplateResponse("home.html", {"request": request})


# ============================================================
# 🤖 Load Toxicity Detection Model
# ============================================================
# FIXED: Only initialize classifier once
try:
    classifier = pipeline("text-classification", model="unitary/toxic-bert")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    classifier = None


# ============================================================
# 🧠 Dashboard (Requires Login)
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main Detection Dashboard"""
    if not request.session.get("user"):
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user": request.session["user"]}
    )


# ============================================================
# 📊 System Statistics API
# ============================================================
# ============================================================
# 📊 System Statistics Endpoint (GET /api/statistics)
# ============================================================
@app.get("/api/statistics")
async def get_statistics():
    """
    Return live system-wide statistics for dashboard display.
    """
    global analysis_count, toxic_count, analysis_history

    if analysis_count == 0:
        return {
            "system_status": "Idle",
            "total_analyses": 0,
            "toxicity_rate": 0,
            "avg_confidence": 0
        }

    # Calculate average confidence and toxicity rate
    avg_confidence = round(sum(a["confidence"] for a in analysis_history) / len(analysis_history), 2)
    toxicity_rate = round((toxic_count / analysis_count) * 100, 2)

    return {
        "system_status": "Online",
        "total_analyses": analysis_count,
        "toxicity_rate": toxicity_rate,
        "avg_confidence": avg_confidence
    }


# ============================================================
# 📥 Request Model Schema
# ============================================================
class TextInput(BaseModel):
    text: str


# ============================================================
# 🔍 Analyze Endpoint (POST /api/analyze)
# ============================================================
@app.post("/api/analyze")
async def analyze_text(input: TextInput):
    """
    Analyze text for toxicity and categorize hate type.
    """
    global analysis_count, toxic_count, analysis_history

    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not available")

    text = input.text.strip()
    if not text:
        return {"error": "No text provided."}

    try:
        result = classifier(text)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    label = result.get("label", "N/A")
    score = float(result.get("score", 0.0))
    confidence = round(score * 100, 2)
    is_toxic = label.lower() in ["toxic", "toxicity", "label_1"]

    # === ✅ Update live stats ===
    analysis_count += 1
    if is_toxic:
        toxic_count += 1

    # Store result for accuracy tracking
    analysis_history.append({"is_toxic": is_toxic, "confidence": confidence})

    # --- Detect Category Type ---
    categories = []
    lowered = text.lower()

    # ============================================================
    # 🔍 Advanced 20-Type Cyberbullying Category Detection
    # ============================================================
    BULLYING_CATEGORIES = {
        "Harassment (Relentless targeting)": [
            "worthless", "trash", "fraud", "liar", "mistake", "delete", "pathetic", "banned",
            "reported", "watching", "targeting", "expose", "harass", "annoying", "useless",
            "block", "hate you", "go away", "no one likes you", "annoy", "irritating", "stalker",
            "blocked", "cancel", "abuse", "harassing", "trolling", "spamming", "disgusting", "obsessed"
        ],
        "Threats & Intimidation": [
            "kill", "hurt", "attack", "ruin", "fire", "careful", "destroy", "scared",
            "find you", "coming for you", "leak", "harm", "burn", "die", "stab",
            "beat", "watch your back", "break you", "hunt", "smash", "wipe out",
            "end you", "threat", "blood", "fear", "revenge", "warn", "burn down", "cut", "explode"
        ],
        "Humiliation & Embarrassment": [
            "embarrassing", "laugh", "photo", "video", "joke", "ashamed", "cry", "recording",
            "pathetic", "humiliate", "weak", "viral", "mock", "clown", "expose",
            "ridicule", "laughingstock", "cringe", "loser", "public shame", "humiliation",
            "shameful", "made fun of", "memes", "screenshots", "roast", "embarrass", "funny pic"
        ],
        "Exclusion & Social Isolation": [
            "not invited", "leave", "alone", "group chat", "ignore", "excluded", "party",
            "unwelcome", "not belong", "without you", "kicked out", "left out", "isolate", "unfriend", "reject",
            "no one cares", "nobody likes you", "don’t come", "not part", "avoid", "stay away",
            "no friends", "forgotten", "left alone", "remove", "outcast", "mute", "blocklist"
        ],
        "Impersonation & Identity Theft": [
            "fake account", "pretending", "catfish", "hack", "impersonate", "using your photo",
            "stolen identity", "parody account", "confession", "fake profile", "post as you",
            "identity theft", "phishing", "spoof", "clone", "copy your posts", "duplicate account",
            "profile theft", "hack account", "pretend to be you", "username copy", "fake messages",
            "posing as you", "account fraud", "use your name", "mirror account", "false identity"
        ],
        "Sexual Harassment": [
            "nude", "slut", "prostitute", "sexual", "easy", "rumor", "private pic",
            "body", "sleep with", "send me", "sex", "put out", "dirty", "inappropriate", "seduce",
            "flirt", "hot", "sexy", "pervert", "touch", "horny", "pics", "x-rated", "lust",
            "babe", "explicit", "bed", "desire", "seductive", "flashing", "NSFW", "rape"
        ],
        "Racial/Ethnic Bullying": [
            "race", "country", "accent", "black", "white", "african", "asian", "arab",
            "foreigner", "immigrant", "culture", "tribe", "slur", "barbaric", "ethnic", "primitive",
            "racist", "colored", "slave", "go back", "third world", "jungle", "monkey", "ethnicity",
            "tribal", "skin color", "heritage", "inferior race", "discrimination", "racism"
        ],
        "Religious Bullying": [
            "religion", "god", "belief", "bible", "quran", "church", "mosque", "terrorist",
            "faith", "cult", "pray", "hindu", "muslim", "christian", "pagan",
            "idol", "atheist", "convert", "hell", "satan", "heaven", "infidel", "unholy", "holy war",
            "religious freak", "fanatic", "blasphemy", "curse", "godless", "sin", "prophet"
        ],
        "Disability Bullying": [
            "retard", "disabled", "wheelchair", "stutter", "special needs", "cripple",
            "limp", "burden", "dumb", "blind", "deaf", "useless", "autistic", "handicapped", "freak",
            "mental", "slow", "spaz", "disease", "abnormal", "broken", "illness", "lame", "mute",
            "invalid", "defective", "weird", "sick", "weak", "handicap", "dumbass"
        ],
        "Sexual Orientation Bullying": [
            "gay", "lesbian", "bisexual", "trans", "queer", "homosexual", "faggot", "abomination",
            "disease", "sin", "unnatural", "attention seeker", "slur", "hell", "deviant",
            "lgbt", "rainbow", "transgender", "drag", "crossdresser", "genderless", "pride", "pervert",
            "homophobic", "closet", "coming out", "disgusting", "gross", "unnatural love"
        ],
        "Gender-Based Bullying": [
            "man", "woman", "girl", "boy", "lady", "weak", "kitchen", "cry", "emotional",
            "aggressive", "sensitive", "blonde", "real man", "inferior", "strong",
            "feminist", "feminazi", "masculine", "girly", "bossy", "submissive", "pretty", "ugly",
            "beta", "alpha", "feminine", "bimbo", "macho", "toxic male", "weak man", "emotional woman"
        ],
        "Body Shaming": [
            "fat", "skinny", "ugly", "acne", "nose", "short", "tall", "greasy",
            "disgusting", "body", "teeth", "weight", "appearance", "hair", "looks",
            "chubby", "fatty", "skeleton", "obese", "belly", "thin", "double chin", "face", "muscles",
            "wrinkles", "shape", "stretch marks", "big nose", "bald", "flat", "too small"
        ],
        "Intellect/Academic Bullying": [
            "stupid", "idiot", "grades", "school", "college", "dumb", "fail", "cheat",
            "class", "math", "writing", "inferior", "education", "illiterate", "dunce",
            "slow learner", "brain dead", "lazy", "zero", "fool", "nerd", "dropout", "clueless",
            "ignorant", "mindless", "illiterate fool", "uneducated", "dumbhead", "failing", "useless brain"
        ],
        "Cyberstalking": [
            "watching", "tracking", "monitor", "online activity", "location", "social media",
            "posts", "following", "footprint", "hack", "screenshot", "spy", "observe", "private", "data",
            "record", "monitoring", "track you", "camera", "surveillance", "geolocation", "creep", "hidden",
            "stalking", "find location", "follow online", "cyber follow", "screenshare", "watchlist"
        ],
        "Reputation Damage": [
            "reputation", "trust", "secret", "liar", "credibility", "rumor", "gossip",
            "name", "community", "hire", "fired", "thief", "slut", "destroyed", "shame",
            "public post", "cancelled", "expose", "fake news", "scandal", "ruined", "bad name", "defame",
            "lying", "false claims", "disgrace", "humiliate", "bad rep", "cancel culture", "insult"
        ],
        "Gaslighting & Manipulation": [
            "overreacting", "sensitive", "crazy", "imagining", "attention", "memory",
            "problem", "joke", "hysterical", "paranoid", "lying", "denying", "making up", "fake", "wrong",
            "you're fine", "not real", "drama", "attention seeker", "insane", "confused", "not true",
            "making things up", "blaming", "control", "mind games", "exaggerating", "overthinking", "gaslight"
        ],
        "Economic/Class Bullying": [
            "poor", "rich", "broke", "car", "neighborhood", "cheap", "trailer", "thrift",
            "background", "money", "class", "status", "embarrassing", "trash", "low-income", "budget",
            "homeless", "unemployed", "middle class", "luxury", "gold digger", "spoiled", "poverty",
            "working class", "ghetto", "billionaire", "poor life", "welfare", "economy", "jobless"
        ],
        "Doxxing & Privacy Invasion": [
            "address", "phone number", "contact", "workplace", "parents", "email",
            "medical", "photo", "leak", "credit card", "password", "search history", "docs", "private info",
            "data leak",
            "post info", "reveal name", "identity", "track", "screenshot", "leaked", "database", "security breach",
            "upload", "expose address", "private data", "information leak", "share contacts", "dox", "gps"
        ],
        "Malicious Reporting": [
            "report", "ban", "remove", "flag", "account", "suspend", "platform", "fake report",
            "mass report", "delete", "privilege", "strike", "complaint", "violation", "policy",
            "false report", "spam report", "targeted flag", "shadowban", "cancel", "admin", "restrict", "post deleted",
            "false ban", "block", "abuse system", "take down", "community report", "fake complaint", "mass flag"
        ],
        "NON BULLYING": [
        "good", "nice", "friend", "happy", "respect", "welcome", "help", "thank you",
        "peace", "kind", "support", "appreciate", "smart", "brilliant", "strong",
        "together", "community", "great job", "love", "positive", "smile"
    ],
    }


    # 🧩 Match text to the best category by frequency of keyword hits
    matched_categories = []
    for cat, words in BULLYING_CATEGORIES.items():
        hits = [w for w in words if w in lowered]
        if hits:
            matched_categories.append((cat, len(hits)))

    if matched_categories:
        # Sort by number of keyword matches (most relevant first)
        matched_categories.sort(key=lambda x: x[1], reverse=True)
        primary_category = matched_categories[0][0]
    else:
        # No matches found — fallback
        primary_category = "Not Bullying / Safe Speech" if not is_toxic else "not bullying"

    # Keep all matched categories (for display)
    categories = [c for c, _ in matched_categories] or [primary_category]

    # ✅ Adjust severity dynamically based on confidence score
    if not is_toxic:
        severity = "low"
    elif score < 0.6:
        severity = "medium"
    elif score < 0.85:
        severity = "high"
    else:
        severity = "critical"

    # ✅ Identify toxic and safe keywords in text for explanation
    toxic_vocab = {
        # Harassment & Insults
        "stupid", "idiot", "worthless", "trash", "garbage", "pathetic", "useless", "fool",
        "liar", "fraud", "moron", "loser", "jerk", "weirdo", "psycho", "freak", "dumb", "crazy",
        "disgusting", "creep", "annoying", "failure", "fraudster", "clown",

        # Threats & Intimidation
        "kill", "hurt", "die", "attack", "destroy", "beat", "harass", "murder", "harm",
        "watch your back", "i know where you live", "i will find you", "you better run",
        "i’m coming for you", "you will pay", "don’t sleep", "i’m watching you", "choke", "stab",

        # Hate Speech / Discrimination
        "terrorist", "extremist", "racist", "black", "white", "muslim", "christian", "jew",
        "hindu", "gay", "lesbian", "trans", "queer", "bisexual", "retard", "cripple",
        "disabled", "immigrant", "foreigner", "refugee", "minority", "colored",

        # Appearance Shaming
        "fat", "ugly", "skinny", "gross", "hideous", "nasty", "stinky", "dirty", "pig", "cow",
        "whale", "bald", "shorty", "giant", "toothless", "wrinkled", "smelly", "hairy",

        # Sexual Harassment
        "slut", "whore", "prostitute", "hoe", "bitch", "sexist", "pervert", "molest", "rape",
        "nude", "porn", "dirty mind", "touch you", "strip", "seduce", "harass you", "sex toy",

        # Cyberbullying / Online Abuse
        "cancel", "block", "reported", "banned", "expose", "trolling", "spamming", "hate you",
        "no one likes you", "get lost", "go away", "nobody cares", "delete yourself",
        "unfollow", "targeting", "worthless account",

        # Academic / Performance Bullying
        "failure", "dropout", "lazy", "slow", "cheater", "fraud", "fake student", "illiterate",
        "uneducated", "dunce", "stupid boy", "stupid girl", "no future", "dumb head",

        # Workplace Bullying
        "fired", "useless worker", "lazy coworker", "incompetent", "deadweight", "office trash",
        "corporate slave", "idiot employee", "worthless staff", "office loser",

        # Social Rejection
        "go away", "nobody likes you", "unwanted", "unloved", "loner", "outcast", "irrelevant",
        "ignored", "hated", "isolated",

        # Family / Personal Abuse
        "bad son", "bad daughter", "terrible mother", "terrible father", "disgrace", "shame",
        "embarrassment", "failure of a family", "you ruin everything",

        # Mental Health Attacks
        "crazy", "insane", "mad", "mental", "depressed freak", "broken mind", "psycho", "sicko",
        "need therapy", "retarded", "unstable", "lunatic",

        # Economic / Class Bullying
        "poor", "broke", "beggar", "useless worker", "cheap", "low class", "trashy", "dirty rich",
        "spoiled", "gold digger",

        # Gender / Sexism
        "weak woman", "crybaby", "feminazi", "beta male", "simp", "manwhore", "playboy",
        "slutty", "emasculated", "useless wife", "dumb girl", "stupid boy",

        # Religious Harassment
        "muslim pig", "christian dog", "terrorist", "infidel", "atheist fool", "fake believer",
        "holy freak", "religious trash",

        # Nationality / Ethnicity Attacks
        "immigrant", "illegal", "foreign trash", "go back to your country", "outsider",
        "unwelcome", "ethnic trash", "third world",

        # Political Harassment
        "corrupt", "traitor", "liar politician", "dictator", "dumb voter", "fake patriot",
        "brainwashed", "sheep", "propaganda fool",

        # Disability & Health Bullying
        "disabled", "handicapped", "cripple", "blind fool", "deaf idiot", "retarded", "sick",
        "weakling", "limp", "diseased", "mutant",

        # Appearance & Fashion
        "ugly clothes", "cheap dress", "dirty look", "no style", "bad haircut", "ugly shoes",
        "smelly perfume", "tacky", "trash outfit",

        # Identity & Social Media Bullying
        "fake account", "fake name", "catfish", "poser", "wannabe", "attention seeker",
        "drama queen", "influencer trash", "content thief", "bot",

        # Misc. General Abuse
        "evil", "creep", "freak", "shame", "nude", "thief", "fraud", "hate", "pig", "jerk"
    }
    safe_vocab = {
        "good", "nice", "friend", "happy", "respect", "welcome", "help", "thank", "peace",
        "kind", "support", "appreciate", "smart", "brilliant", "strong", "together", "community",
        "great", "love", "positive", "smile", "excellent", "hero", "talented", "amazing", "awesome",
        "cool", "well done", "best", "beautiful", "helpful", "calm", "generous"
    }

    # ✅ Detect both toxic and safe hits
    toxic_hits = [word for word in toxic_vocab if word in lowered]
    safe_hits = [word for word in safe_vocab if word in lowered]

    # ✅ Refine result based on hits
    if toxic_hits and not safe_hits:
        final_label = "toxic"
        explanation = f"Toxic content detected: {', '.join(toxic_hits)}"
        suggestion = ["Avoid threats, harassment, or hate speech", "Use respectful and inclusive language"]
    elif safe_hits and not toxic_hits:
        final_label = "non-toxic"
        is_toxic = False
        explanation = f"Positive / safe words detected: {', '.join(safe_hits)}"
        suggestion = ["Positive and respectful tone maintained"]
        score = 0.0
        confidence = 100.0
    elif toxic_hits and safe_hits:
        final_label = "mixed"
        is_toxic = False
        explanation = (
            f"Both toxic ({', '.join(toxic_hits)}) and safe ({', '.join(safe_hits)}) words detected. "
            "Context may alter meaning."
        )
        suggestion = ["Review text manually — contains conflicting tone"]
        score = 0.4
    else:
        final_label = label.lower()
        explanation = "No toxic or safe keywords detected."
        suggestion = ["Neutral / context-safe language used"]

    # --- ✅ Return dynamic response ---
    return {
        "text": text,
        "prediction": final_label,
        "confidence": confidence,
        "score": round(score, 2),
        "is_toxic": final_label == "toxic",
        "severity": severity,
        "primary_category": primary_category,
        "categories": categories,
        "explanation": explanation,
        "suggestions": suggestion,
        "detailed_analysis": {
            "pattern_matches": categories,
            "toxic_words": toxic_hits or ["None"],
            "safe_words": safe_hits or ["None"]
        },
    }
# ============================================================
# 🎯 Health Check Endpoint
# ============================================================
@app.get("/health")
async def health_check():
    """Check if API is running"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "total_analyses": analysis_count
    }