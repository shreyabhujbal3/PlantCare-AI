import os, sqlite3, uuid
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-before-deployment")
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED = {"png", "jpg", "jpeg", "webp"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
DB = "plantcare.db"
_model = _processor = None

def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db() as con:
        con.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL)")
        con.execute("""CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, image TEXT NOT NULL, plant TEXT, disease TEXT NOT NULL, confidence REAL, created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id))""")
init_db()

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "info")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def lang():
    return session.get("lang", "en")

@app.context_processor
def inject():
    return {"lang": lang(), "logged_in": "user_id" in session}

def load_model():
    global _model, _processor
    if _model is None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        model_id = os.environ.get("PLANT_MODEL_ID", "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
        _processor = AutoImageProcessor.from_pretrained(model_id)
        _model = AutoModelForImageClassification.from_pretrained(model_id)
        _model.eval()
    return _processor, _model

def predict(path):
    from PIL import Image
    import torch
    processor, model = load_model()
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        index = int(torch.argmax(probabilities))
    return model.config.id2label.get(index, f"Class {index}").replace("_", " "), float(probabilities[index]) * 100

def disease_details(label, language):
    mr = language == "mr"
    s = label.lower()
    if "scorch" in s or "blight" in s or "spot" in s:
        if mr:
            return {"summary":"पानांवरील तपकिरी/काळे डाग किंवा वाळलेले भाग या लेबलाशी जुळू शकतात; पण पाणी, उष्णता, पोषणातील असमतोल किंवा इतर रोगांमुळेही अशी लक्षणे दिसू शकतात.",
            "symptoms":["पानांवर तपकिरी किंवा काळे डाग","पानांच्या कडा वाळणे किंवा पिवळेपणा","गंभीर स्थितीत पानाचा काही भाग सुकणे"],
            "causes":["रोगकारक किंवा बुरशीजन्य समस्या शक्य","उष्णता, पाण्याचा ताण किंवा पोषणातील असमतोलही शक्य"],
            "prevention":["मातीतील ओलावा आणि निचरा तपासा","झाडांमध्ये हवा खेळती ठेवा; साधने स्वच्छ करा","स्थानिक कृषी तज्ज्ञांकडून खात्री करूनच मंजूर उपचार वापरा"],
            "photo_tips":["पानाच्या दोन्ही बाजूंचा आणि संपूर्ण झाडाचा फोटो घ्या","लक्षणांचा स्पष्ट जवळचा फोटो घ्या","पिकाचे नाव व लक्षणे कधी सुरू झाली ते नोंदवा"]}
        return {"summary":"Brown/dark spots or dry areas can resemble this label, but heat, water stress, nutrition issues, and other diseases may look similar.",
        "symptoms":["Brown or dark leaf spots","Dry leaf edges or yellowing","Severe cases may dry parts of the leaf"],
        "causes":["A disease agent may be involved","Heat, water stress, or nutrient imbalance may look similar"],
        "prevention":["Check soil moisture and drainage","Improve airflow and clean tools","Confirm locally before using an approved treatment"],
        "photo_tips":["Photograph both sides of the leaf and the whole plant","Take a sharp close-up of symptoms","Record crop name and when symptoms began"]}
    if mr:
        return {"summary":"AI लेबल हा फोटोवर आधारित प्राथमिक अंदाज आहे, निश्चित निदान नाही.",
        "symptoms":["फोटोमध्ये रंगबदल, डाग किंवा वाळलेले भाग असू शकतात; प्रत्यक्ष तपासणी आवश्यक आहे."],
        "causes":["बुरशी, जीवाणू, विषाणू, कीड, पाणी किंवा पोषणातील समस्या सारखी दिसू शकतात."],
        "prevention":["पाणी, निचरा, सूर्यप्रकाश आणि किडी तपासा.","बाधित भाग हाताळल्यानंतर साधने स्वच्छ करा.","रासायनिक उपचारापूर्वी कृषी अधिकारी/KVK चा सल्ला घ्या."],
        "photo_tips":["पानाच्या दोन्ही बाजू आणि संपूर्ण झाडाचे फोटो घ्या.","पिकाचे नाव आणि लक्षणे सुरू झाल्याची तारीख नोंदवा."]}
    return {"summary":"This AI label is a preliminary image-based guess, not a confirmed diagnosis.",
        "symptoms":["The photo may show discoloration, spots, or dried areas; an in-person check is needed."],
        "causes":["Fungal, bacterial, viral, insect, water, and nutrient problems can look similar."],
        "prevention":["Check watering, drainage, sunlight, and insects.","Clean tools after handling affected plants.","Consult an agriculture officer/KVK before chemical treatment."],
        "photo_tips":["Photograph both leaf sides and the whole plant.","Record crop name and when symptoms began."]}

def nearby_support(language, area):
    if language == "mr":
        return [
        {"name":"जवळचे कृषी सेवा केंद्र","detail":"औषध/खत घेण्यापूर्वी पिकाचे नाव व अहवाल दाखवा; साठा फोन करून तपासा.","query":"Krishi Seva Kendra near " + area},
        {"name":"तालुका कृषी अधिकारी / KVK","detail":"रोगाची खात्री आणि पीकनिहाय मार्गदर्शनासाठी.","query":"Agriculture office Krishi Vigyan Kendra near " + area},
        {"name":"खते व पीक संरक्षण दुकान","detail":"पिकासाठी मंजूर उत्पादन आणि लेबलवरील सूचना तपासा.","query":"fertilizer pesticide shop near " + area}]
    return [
        {"name":"Nearby Krishi Seva Kendra","detail":"Show the crop and report; call to confirm stock before travelling.","query":"Krishi Seva Kendra near " + area},
        {"name":"Taluka Agriculture Office / KVK","detail":"For diagnosis confirmation and crop-specific advice.","query":"Agriculture office Krishi Vigyan Kendra near " + area},
        {"name":"Fertilizer & crop-protection shop","detail":"Confirm the product is approved for your crop and follow its label.","query":"fertilizer pesticide shop near " + area}]

def advice(label, language):
    s = label.lower()
    if "healthy" in s:
        en = ["The model classified this leaf as healthy. Monitor new growth regularly.", "Water at the base and avoid prolonged leaf wetness.", "Maintain suitable spacing and use fertilizer according to soil needs."]
        mr = ["मॉडेलनुसार हे पान निरोगी दिसत आहे. नवीन वाढीचे नियमित निरीक्षण करा.", "झाडाच्या मुळाशी पाणी द्या आणि पानांवर सतत पाणी राहू देऊ नका.", "योग्य अंतर ठेवा आणि मातीच्या गरजेनुसार खत वापरा."]
    elif any(x in s for x in ["blight"]):
        en = ["Remove badly affected leaves and dispose of them safely.", "Avoid overhead watering and improve airflow between plants.", "Confirm the disease locally before using any crop-approved treatment."]
        mr = ["जास्त बाधित पाने काढून सुरक्षितरीत्या नष्ट करा.", "वरून पाणी देणे टाळा आणि झाडांमध्ये हवा खेळती ठेवा.", "कोणतेही औषध वापरण्यापूर्वी स्थानिक कृषी तज्ज्ञांकडून रोगाची खात्री करा."]
    elif any(x in s for x in ["rust", "mildew", "mold", "mould"]):
        en = ["Remove affected leaves where practical and clean tools.", "Keep foliage dry and improve airflow.", "Use only a registered treatment according to its label after confirming the diagnosis."]
        mr = ["शक्य असल्यास बाधित पाने काढा आणि साधने स्वच्छ करा.", "पाने कोरडी राहतील याची काळजी घ्या आणि हवा खेळती ठेवा.", "रोगाची खात्री करूनच लेबलनुसार मंजूर औषध वापरा."]
    elif any(x in s for x in ["virus", "mosaic", "yellow leaf curl"]):
        en = ["Separate suspicious plants and disinfect tools.", "Check for insects such as aphids or whiteflies; avoid moving infected seedlings.", "Ask an agricultural expert to confirm the diagnosis."]
        mr = ["संशयित झाडे वेगळी ठेवा आणि साधने निर्जंतुक करा.", "मावा किंवा पांढऱ्या माशीसारख्या किडी तपासा; बाधित रोपे हलवू नका.", "कृषी तज्ज्ञांकडून रोगाची खात्री करून घ्या."]
    else:
        en = ["Remove clearly affected plant parts and dispose of them safely.", "Check watering, drainage, sunlight, and pest activity.", "This is an AI screening result, not a confirmed diagnosis. Consult a local agriculture officer before chemical treatment."]
        mr = ["स्पष्टपणे बाधित भाग काढून सुरक्षितरीत्या नष्ट करा.", "पाणी, निचरा, सूर्यप्रकाश आणि किडींची तपासणी करा.", "हा AI चा प्राथमिक अंदाज आहे, निश्चित निदान नाही. रासायनिक उपचारापूर्वी कृषी अधिकाऱ्यांचा सल्ला घ्या."]
    return mr if language == "mr" else en

def shops(language):
    if language == "mr":
        return [("स्थानिक कृषी सेवा केंद्र", "औषध घेण्यापूर्वी AI अहवाल दाखवा."), ("तालुका कृषी अधिकारी", "रोगाची खात्री व योग्य उपायांसाठी."), ("कृषी विज्ञान केंद्र (KVK)", "पीक संरक्षणाबाबत तज्ज्ञ मार्गदर्शनासाठी.")]
    return [("Local Krishi Seva Kendra", "Show the report before purchasing products."), ("Taluka Agriculture Office", "For diagnosis confirmation and local advice."), ("Krishi Vigyan Kendra (KVK)", "For expert crop-protection guidance.")]

@app.route("/")
def index():
    return redirect(url_for("dashboard") if "user_id" in session else url_for("login"))

@app.route("/language/<code>")
def language(code):
    if code in ("en", "mr"): session["lang"] = code
    return redirect(request.referrer or url_for("index"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name, email, password = request.form.get("name", "").strip(), request.form.get("email", "").strip().lower(), request.form.get("password", "")
        if not name or not email or len(password) < 6:
            flash("Enter all fields. Password must be at least 6 characters.", "error")
        else:
            try:
                with db() as con:
                    con.execute("INSERT INTO users(name,email,password) VALUES(?,?,?)", (name, email, generate_password_hash(password)))
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("This email is already registered.", "error")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email, password = request.form.get("email", "").strip().lower(), request.form.get("password", "")
        with db() as con: user = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password"], password):
            session["user_id"], session["user_name"] = user["id"], user["name"]
            return redirect(url_for("dashboard"))
        flash("Incorrect email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", name=session.get("user_name", "Farmer"))

@app.route("/detect", methods=["GET", "POST"])
@login_required
def detect():
    result = None
    if request.method == "POST":
        file = request.files.get("plant_image")
        if not file or not file.filename:
            flash("Please choose a plant image.", "error")
        else:
            ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
            if ext not in ALLOWED:
                flash("Use JPG, JPEG, PNG, or WEBP.", "error")
            else:
                filename = uuid.uuid4().hex + "_" + secure_filename(file.filename)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)
                try:
                    disease, confidence = predict(path)
                    plant = disease.split("___")[0].replace("_", " ") if "___" in disease else "Plant / leaf"
                    with db() as con:
                        con.execute("INSERT INTO scans(user_id,image,plant,disease,confidence,created_at) VALUES(?,?,?,?,?,?)",
                            (session["user_id"], filename, plant, disease, confidence, datetime.now().strftime("%d %b %Y, %I:%M %p")))
                    result = {"image": filename, "plant": plant, "disease": disease, "confidence": round(confidence, 1),
                              "advice": advice(disease, lang()), "details": disease_details(disease, lang()),
                              "shops": nearby_support(lang(), request.form.get("area", "").strip() or "your area")}
                except Exception as e:
                    try: os.remove(path)
                    except OSError: pass
                    flash("AI model could not run. Check internet/model setup. Details: " + str(e), "error")
    return render_template("detect.html", result=result)

@app.route("/history")
@login_required
def history():
    with db() as con:
        rows = con.execute("SELECT * FROM scans WHERE user_id=? ORDER BY id DESC", (session["user_id"],)).fetchall()
    return render_template("history.html", rows=rows)

@app.route("/about")
def about():
    return render_template("about.html")

@app.errorhandler(413)
def too_large(error):
    flash("Image is too large. Maximum size is 10 MB.", "error")
    return redirect(url_for("detect"))

if __name__ == "__main__":
    app.run(debug=True)
