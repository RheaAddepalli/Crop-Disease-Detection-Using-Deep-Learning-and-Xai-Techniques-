import tensorflow as tf
import numpy as np
import cv2
import os

# ============================================================
# PATH SETUP
# ============================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# LOAD MODEL
# ============================================================
model = tf.keras.models.load_model(
    os.path.join(ROOT_DIR, "src", "models", "best_crop_disease_model.keras")
)
print("✔ Model loaded")

# ============================================================
# CLASS NAMES
# ============================================================
dataset_dir = os.path.join(ROOT_DIR, "datasets", "train_dataset")

class_names = sorted([
    d for d in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, d))
])
print("✔ Classes:", class_names)

# ============================================================
# LAST CONV LAYER
# ============================================================
last_conv_name = "block_16_project"
last_conv_layer = model.get_layer(last_conv_name)

# ============================================================
# CONFIG (TUNE THESE)
# ============================================================
CONF_THRESHOLD = 0.6      # main confidence threshold
ENTROPY_THRESHOLD = 1.5   # higher = more uncertain
GAP_THRESHOLD = 0.15      # difference between top1 & top2

# ============================================================
# GRAD-CAM + PREDICTION
# ============================================================
def generate_gradcam(image_path):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_array)

        preds = predictions[0].numpy()

        pred_index = np.argmax(preds)
        confidence = np.max(preds)

        # ==========================
        # UNCERTAINTY METRICS
        # ==========================
        # Entropy
        entropy = -np.sum(preds * np.log(preds + 1e-9))

        # Top-2 gap
        top2 = np.sort(preds)[-2:]
        gap = top2[1] - top2[0]

        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_out)[0]
    conv_out = conv_out[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)

    for i in range(weights.shape[0]):
        heatmap += weights[i] * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-9)

    original = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    result = cv2.addWeighted(original, 0.3, heatmap, 0.7, 0)

    return pred_index, confidence, entropy, gap, result


# ============================================================
# FINAL PREDICTION FUNCTION
# ============================================================
def predict_for_web(image_path):
    idx, confidence, entropy, gap, grad_img = generate_gradcam(image_path)

    predicted_class = class_names[idx]

    # ==========================
    # REJECTION LOGIC
    # ==========================
    if confidence < CONF_THRESHOLD:
        return {
            "label": "❌ Unknown / Not a crop disease",
            "confidence": float(confidence),
            "reason": "Low confidence",
            "entropy": float(entropy),
            "gap": float(gap),
            "image": grad_img
        }

    if entropy > ENTROPY_THRESHOLD:
        return {
            "label": "❌ Uncertain prediction",
            "confidence": float(confidence),
            "reason": "High entropy (model confused)",
            "entropy": float(entropy),
            "gap": float(gap),
            "image": grad_img
        }

    if gap < GAP_THRESHOLD:
        return {
            "label": "❌ Ambiguous result",
            "confidence": float(confidence),
            "reason": "Top classes too close",
            "entropy": float(entropy),
            "gap": float(gap),
            "image": grad_img
        }

    # ==========================
    # ACCEPTED PREDICTION
    # ==========================
    return {
        "label": predicted_class,
        "confidence": float(confidence),
        "reason": "High confidence prediction",
        "entropy": float(entropy),
        "gap": float(gap),
        "image": grad_img
    }









# # without confidence
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8"/>
# <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
# <title>CropScan — Hey, leaf got issues?</title>
# <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
# <style>
#   :root {
#     --green:  #00e676;
#     --teal:   #1de9b6;
#     --lime:   #c6ff00;
#     --dark:   #0a0f0a;
#     --card:   #111711;
#     --border: #1e2e1e;
#     --text:   #e8f5e9;
#     --muted:  #5a7a5a;
#   }
#   *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
#   body {
#     background: var(--dark);
#     color: var(--text);
#     font-family: 'DM Sans', sans-serif;
#     min-height: 100vh;
#     overflow-x: hidden;
#   }
#   body::before {
#     content: '';
#     position: fixed; inset: 0;
#     background-image:
#       linear-gradient(rgba(0,230,118,.04) 1px, transparent 1px),
#       linear-gradient(90deg, rgba(0,230,118,.04) 1px, transparent 1px);
#     background-size: 40px 40px;
#     pointer-events: none; z-index: 0;
#   }
#   .blob { position:fixed; border-radius:50%; filter:blur(100px); opacity:0.13; pointer-events:none; z-index:0; animation:drift 12s ease-in-out infinite alternate; }
#   .blob1 { width:500px; height:500px; background:#00e676; top:-150px; left:-150px; }
#   .blob2 { width:400px; height:400px; background:#1de9b6; bottom:-100px; right:-100px; animation-delay:3s; }
#   .blob3 { width:300px; height:300px; background:#c6ff00; top:50%; left:40%; animation-delay:6s; }
#   @keyframes drift { from{transform:translate(0,0) scale(1)} to{transform:translate(30px,20px) scale(1.08)} }

#   .page {
#     position: relative; z-index: 1;
#     display: grid;
#     grid-template-columns: 1fr 290px;
#     grid-template-rows: auto 1fr;
#     min-height: 100vh;
#   }

#   /* HEADER */
#   header {
#     grid-column: 1 / -1;
#     padding: 22px 36px;
#     display: flex; align-items: center; gap: 14px;
#     border-bottom: 1px solid var(--border);
#     background: rgba(10,15,10,.8);
#     backdrop-filter: blur(12px);
#   }
#   .logo { width:40px; height:40px; background:linear-gradient(135deg,var(--green),var(--teal)); border-radius:11px; display:flex; align-items:center; justify-content:center; font-size:19px; }
#   header h1 { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; background:linear-gradient(90deg,var(--green),var(--teal)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
#   .live { margin-left:auto; display:flex; align-items:center; gap:8px; font-size:0.75rem; color:var(--muted); }
#   .dot { width:8px; height:8px; background:var(--green); border-radius:50%; box-shadow:0 0 8px var(--green); animation:pulse 2s infinite; }
#   @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

#   /* MAIN */
#   main { padding:36px; display:flex; flex-direction:column; gap:0; overflow-y:auto; }

#   /* CHAT BUBBLES */
#   .chat { display:flex; flex-direction:column; gap:18px; }

#   .bubble-wrap { display:flex; align-items:flex-end; gap:10px; animation:fadeUp .5s ease; }
#   .bubble-wrap.right { flex-direction:row-reverse; }

#   @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }

#   .avatar {
#     width:36px; height:36px; border-radius:50%; flex-shrink:0;
#     display:flex; align-items:center; justify-content:center; font-size:18px;
#     background: linear-gradient(135deg,#0d2a0d,#1a3d1a);
#     border: 1px solid var(--border);
#   }
#   .avatar.user { background:linear-gradient(135deg,#1a1a2e,#2d2d4e); }

#   .bubble {
#     max-width: 520px;
#     padding: 14px 18px;
#     border-radius: 18px;
#     font-size: 0.92rem;
#     line-height: 1.6;
#     border: 1px solid var(--border);
#   }
#   .bubble.bot {
#     background: var(--card);
#     border-radius: 18px 18px 18px 4px;
#     color: var(--text);
#   }
#   .bubble.user {
#     background: linear-gradient(135deg, rgba(0,230,118,.12), rgba(29,233,182,.08));
#     border-color: rgba(0,230,118,.25);
#     border-radius: 18px 18px 4px 18px;
#     color: var(--text);
#   }
#   .bubble strong { color: var(--lime); font-weight:700; }
#   .bubble .highlight { color: var(--green); font-weight:600; }

#   /* CHIPS for crop types */
#   .crop-chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
#   .crop-chip {
#     padding:6px 14px; border-radius:20px; font-size:0.8rem; font-weight:600;
#     border:1px solid var(--border); background:rgba(255,255,255,.04);
#     cursor:default;
#   }
#   .crop-chip.corn   { border-color:#f9a825; color:#f9a825; }
#   .crop-chip.potato { border-color:#ab47bc; color:#ab47bc; }
#   .crop-chip.tomato { border-color:#ef5350; color:#ef5350; }

#   /* UPLOAD AREA inside chat */
#   .upload-bubble {
#     max-width: 520px;
#     background: var(--card);
#     border: 2px dashed var(--border);
#     border-radius: 18px 18px 18px 4px;
#     padding: 22px;
#     cursor:pointer;
#     transition: border-color .3s, background .3s;
#     position: relative;
#   }
#   .upload-bubble:hover { border-color:var(--green); background:rgba(0,230,118,.04); }
#   .upload-bubble input { position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%; }
#   .upload-bubble p { font-size:0.85rem; color:var(--muted); margin-top:6px; }

#   /* PREVIEW inside chat */
#   .preview-bubble {
#     max-width:300px;
#     border-radius:14px;
#     overflow:hidden;
#     border:1px solid var(--border);
#   }
#   .preview-bubble img { width:100%; display:block; max-height:260px; object-fit:cover; }
#   .preview-label { padding:8px 12px; background:var(--card); font-size:0.78rem; color:var(--muted); }

#   /* ANALYSE BUTTON */
#   .btn-analyse {
#     align-self:flex-start;
#     padding:13px 28px; border:none; outline:none; cursor:pointer;
#     border-radius:14px; font-family:'Syne',sans-serif;
#     font-size:0.95rem; font-weight:700; letter-spacing:.5px;
#     background:linear-gradient(135deg,var(--green),var(--teal));
#     color:#0a0f0a;
#     box-shadow:0 0 24px rgba(0,230,118,.25);
#     transition:transform .2s,box-shadow .2s,opacity .2s;
#   }
#   .btn-analyse:hover { transform:translateY(-2px); box-shadow:0 0 40px rgba(0,230,118,.4); }
#   .btn-analyse:disabled { opacity:.5; cursor:not-allowed; transform:none; }

#   /* RESULT DISEASE NAME */
#   .disease-name {
#     font-family:'Syne',sans-serif;
#     font-size:1.5rem; font-weight:800;
#     color:var(--lime);
#     margin: 6px 0 4px;
#   }

#   /* GRADCAM reveal button */
#   .btn-gradcam {
#     margin-top:12px;
#     padding:10px 20px; border:1px solid rgba(0,230,118,.3); outline:none; cursor:pointer;
#     border-radius:10px; font-family:'DM Sans',sans-serif;
#     font-size:0.85rem; font-weight:500;
#     background:rgba(0,230,118,.07); color:var(--green);
#     transition:background .2s, border-color .2s;
#   }
#   .btn-gradcam:hover { background:rgba(0,230,118,.14); border-color:var(--green); }

#   /* GRADCAM IMAGE */
#   .gradcam-wrap {
#     max-width:380px;
#     border-radius:14px; overflow:hidden;
#     border:1px solid var(--border);
#     margin-top:4px;
#   }
#   .gradcam-wrap img { width:100%; display:block; }
#   .gradcam-caption { padding:8px 12px; background:var(--card); font-size:0.75rem; color:var(--muted); }

#   /* SPINNER */
#   .spinner { display:none; width:24px; height:24px; border:3px solid var(--border); border-top-color:var(--green); border-radius:50%; animation:spin .7s linear infinite; }
#   .spinner.show { display:inline-block; }
#   @keyframes spin { to{transform:rotate(360deg)} }

#   .typing { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:0.82rem; }
#   .typing span { display:inline-block; width:6px; height:6px; background:var(--muted); border-radius:50%; animation:bounce 1.2s infinite; }
#   .typing span:nth-child(2) { animation-delay:.2s; }
#   .typing span:nth-child(3) { animation-delay:.4s; }
#   @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)} }

#   /* HIDDEN */
#   .hidden { display:none !important; }

#   /* SIDEBAR */
#   aside { border-left:1px solid var(--border); background:rgba(10,15,10,.5); backdrop-filter:blur(10px); overflow-y:auto; padding:26px 18px; display:flex; flex-direction:column; gap:26px; }
#   aside::-webkit-scrollbar { width:4px; }
#   aside::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
#   .slabel { font-family:'Syne',sans-serif; font-size:0.65rem; font-weight:700; letter-spacing:3px; text-transform:uppercase; color:var(--green); margin-bottom:10px; }

#   /* SAMPLES */
#   .sample-grid { display:flex; flex-direction:column; gap:9px; }
#   .sample-card { border-radius:11px; overflow:hidden; border:1px solid var(--border); cursor:pointer; transition:border-color .2s,transform .2s; }
#   .sample-card:hover { border-color:var(--green); transform:translateX(4px); }
#   .sample-img { width:100%; height:76px; object-fit:cover; display:block; }
#   .sample-img-ph { width:100%; height:76px; background:linear-gradient(135deg,#0d1f0d,#162516); display:flex; align-items:center; justify-content:center; font-size:1.6rem; }
#   .sample-name { padding:6px 10px; font-size:0.68rem; color:var(--muted); background:var(--card); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

#   /* CLASSES */
#   .classes-list { display:flex; flex-direction:column; gap:5px; }
#   .chip { display:flex; align-items:center; gap:8px; padding:7px 10px; border-radius:8px; border:1px solid var(--border); background:rgba(255,255,255,.02); font-size:0.7rem; color:var(--text); transition:background .2s,border-color .2s; }
#   .chip:hover { background:rgba(0,230,118,.05); border-color:rgba(0,230,118,.2); }
#   .cdot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
#   .ctag { font-size:0.6rem; color:var(--muted); margin-left:auto; flex-shrink:0; }
#   .corn   { background:#f9a825; }
#   .potato { background:#ab47bc; }
#   .tomato { background:#ef5350; }
# </style>
# </head>
# <body>

# <div class="blob blob1"></div>
# <div class="blob blob2"></div>
# <div class="blob blob3"></div>

# <div class="page">

#   <header>
#     <div class="logo">🌿</div>
#     <h1>CropScan</h1>
#     <div class="live"><div class="dot"></div>Online</div>
#   </header>

#   <main>
#     <div class="chat" id="chat">

#       <!-- BOT: Opening message -->
#       <div class="bubble-wrap" id="msgWelcome">
#         <div class="avatar">🤖</div>
#         <div class="bubble bot">
#           Hey! 👋 Got some problem with your crop leaf? <strong>Let me help!</strong><br><br>
#           I'm trained on <span class="highlight">17 classes</span> across these crops — so make sure you pick a
#           <strong>Tomato</strong>, <strong>Corn</strong>, or <strong>Potato</strong> leaf, buddy! 🌱
#           <div class="crop-chips">
#             <span class="crop-chip tomato">🍅 Tomato</span>
#             <span class="crop-chip corn">🌽 Corn</span>
#             <span class="crop-chip potato">🥔 Potato</span>
#           </div>
#         </div>
#       </div>

#       <!-- BOT: Upload prompt -->
#       <div class="bubble-wrap" id="msgUploadPrompt">
#         <div class="avatar">🤖</div>
#         <div class="upload-bubble" id="uploadZone">
#           <input type="file" id="imageInput" accept="image/*" onchange="onImagePicked()"/>
#           <div style="font-size:1.8rem">📸</div>
#           <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;margin-top:8px;">Drop your leaf photo here</div>
#           <p>or click to browse — JPG / PNG</p>
#         </div>
#       </div>

#       <!-- USER: Image preview — hidden until image picked -->
#       <div class="bubble-wrap right hidden" id="msgPreview">
#         <div class="avatar user">👤</div>
#         <div>
#           <div class="preview-bubble">
#             <img id="previewImg" alt="Leaf preview"/>
#             <div class="preview-label" id="previewLabel">Your leaf</div>
#           </div>
#         </div>
#       </div>

#       <!-- BOT: Analyse button — hidden until image picked -->
#       <div class="bubble-wrap hidden" id="msgAnalysePrompt">
#         <div class="avatar">🤖</div>
#         <div class="bubble bot">
#           Nice! Got your leaf 🍃 Let me take a closer look at it.<br>
#           Hit the button and I'll tell you what's going on!
#         </div>
#       </div>

#       <div class="bubble-wrap hidden" id="msgAnalyseBtn">
#         <div class="avatar" style="opacity:0">x</div>
#         <button class="btn-analyse" id="analyseBtn" onclick="runPredict()">🔍 Analyse This Leaf</button>
#       </div>

#       <!-- BOT: Typing indicator — hidden until predicting -->
#       <div class="bubble-wrap hidden" id="msgTyping">
#         <div class="avatar">🤖</div>
#         <div class="bubble bot">
#           <div class="typing"><span></span><span></span><span></span></div>
#         </div>
#       </div>

#       <!-- BOT: Result — disease name first -->
#       <div class="bubble-wrap hidden" id="msgResult">
#         <div class="avatar">🤖</div>
#         <div class="bubble bot">
#           Looks like this leaf is —
#           <div class="disease-name" id="diseaseName">—</div>
#           <span id="resultEmoji" style="font-size:1.3rem"></span>
#         </div>
#       </div>

#       <!-- BOT: Confirmation + GradCAM offer in one block -->
#       <div class="bubble-wrap hidden" id="msgConfirm">
#         <div class="avatar">🤖</div>
#         <div class="bubble bot">
#           Ok! I think now you know what is the <strong>leaf status</strong>, right? 🎯<br><br>
#           You also want to know <strong>on what areas I focused on</strong>? 👀
#           I can show you exactly where I was looking on the leaf!
#           <div>
#             <button class="btn-gradcam" id="gradcamOfferBtn" onclick="showGradcam()">🔥 Yes, show me the heatmap!</button>
#           </div>
#         </div>
#       </div>

#       <!-- BOT: GradCAM image — hidden until clicked -->
#       <div class="bubble-wrap hidden" id="msgGradcam">
#         <div class="avatar">🤖</div>
#         <div>
#           <div class="bubble bot" style="margin-bottom:10px">
#             Here you go! 🔥 The <span class="highlight">red/orange zones</span> are where I focused the most —
#             that's what gave me the prediction!
#           </div>
#           <div class="gradcam-wrap">
#             <img id="gradcamImg" alt="Grad-CAM heatmap"/>
#             <div class="gradcam-caption">🌡 Grad-CAM Heatmap — hot zones = high attention areas</div>
#           </div>
#         </div>
#       </div>

#       <!-- BOT: Try again -->
#       <div class="bubble-wrap hidden" id="msgTryAgain">
#         <div class="avatar">🤖</div>
#         <div class="bubble bot">
#           Want to check another leaf? Just drop a new image above anytime! 🌿
#         </div>
#       </div>

#     </div>
#   </main>

#   <!-- SIDEBAR -->
#   <aside>

#     <div>
#       <div class="slabel">Sample Images</div>
#       <div class="sample-grid" id="sampleGrid">
#         <div style="color:var(--muted);font-size:0.75rem">Loading…</div>
#       </div>
#     </div>

#     <div>
#       <div class="slabel">17 Classes</div>
#       <div class="classes-list" id="classesList"></div>
#     </div>

#   </aside>

# </div>

# <script>
# // ── State ──
# let gradcamUrl = null;

# // ── Classes ──
# const CLASSES = [
#   { name:"Corn — Cercospora / Gray Leaf Spot", crop:"Corn" },
#   { name:"Corn — Common Rust",                  crop:"Corn" },
#   { name:"Corn — Healthy",                      crop:"Corn" },
#   { name:"Corn — Northern Leaf Blight",         crop:"Corn" },
#   { name:"Potato — Early Blight",               crop:"Potato" },
#   { name:"Potato — Healthy",                    crop:"Potato" },
#   { name:"Potato — Late Blight",                crop:"Potato" },
#   { name:"Tomato — Bacterial Spot",             crop:"Tomato" },
#   { name:"Tomato — Early Blight",               crop:"Tomato" },
#   { name:"Tomato — Healthy",                    crop:"Tomato" },
#   { name:"Tomato — Late Blight",                crop:"Tomato" },
#   { name:"Tomato — Leaf Mold",                  crop:"Tomato" },
#   { name:"Tomato — Septoria Leaf Spot",         crop:"Tomato" },
#   { name:"Tomato — Spider Mites",               crop:"Tomato" },
#   { name:"Tomato — Target Spot",                crop:"Tomato" },
#   { name:"Tomato — Mosaic Virus",               crop:"Tomato" },
#   { name:"Tomato — Yellow Leaf Curl Virus",     crop:"Tomato" },
# ];
# const EMOJI  = { Corn:"🌽", Potato:"🥔", Tomato:"🍅" };
# const COLOR  = { Corn:"corn", Potato:"potato", Tomato:"tomato" };
# const RESULT_EMOJI = {
#   "healthy": "✅", "Healthy": "✅",
#   "rust": "🟤", "blight": "🔴", "Blight": "🔴",
#   "spot": "🟡", "mold": "🟠", "mites": "🔴",
#   "virus": "🦠", "Virus": "🦠",
# };

# function getResultEmoji(name) {
#   for (const [key, val] of Object.entries(RESULT_EMOJI)) {
#     if (name.toLowerCase().includes(key.toLowerCase())) return val;
#   }
#   return "🌿";
# }

# // Build classes list
# document.getElementById("classesList").innerHTML = CLASSES.map(c => `
#   <div class="chip">
#     <div class="cdot ${COLOR[c.crop]}"></div>
#     <span>${c.name}</span>
#     <span class="ctag">${EMOJI[c.crop]}</span>
#   </div>
# `).join("");

# // ── Load samples ──
# fetch("/samples")
#   .then(r => r.json())
#   .then(data => {
#     const grid = document.getElementById("sampleGrid");
#     if (!data.samples || !data.samples.length) {
#       grid.innerHTML = `<div style="color:var(--muted);font-size:0.75rem">Add images to<br>dataset/sample_web_images/</div>`;
#       return;
#     }
#     const picks = data.samples.sort(() => Math.random() - 0.5).slice(0, 3);
#     grid.innerHTML = picks.map(f => `
#       <div class="sample-card" onclick="loadSample('${f}')">
#         <img class="sample-img" src="/samples/${f}" alt="${f}"
#           onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"/>
#         <div class="sample-img-ph" style="display:none">🌿</div>
#         <div class="sample-name">${f}</div>
#       </div>
#     `).join("");
#   })
#   .catch(() => {
#     document.getElementById("sampleGrid").innerHTML =
#       `<div style="color:var(--muted);font-size:0.75rem">Could not load samples.</div>`;
#   });

# function loadSample(filename) {
#   fetch(`/samples/${filename}`)
#     .then(r => r.blob())
#     .then(blob => {
#       const file = new File([blob], filename, { type: blob.type });
#       const dt = new DataTransfer();
#       dt.items.add(file);
#       document.getElementById("imageInput").files = dt.files;
#       onImagePicked();
#     });
# }

# // ── Image picked ──
# function onImagePicked() {
#   const input = document.getElementById("imageInput");
#   if (!input.files.length) return;
#   const file = input.files[0];

#   // Reset all previous state
#   hide("msgResult"); hide("msgConfirm"); hide("msgGradcam"); hide("msgTryAgain");
#   hide("msgAnalysePrompt"); hide("msgAnalyseBtn");
#   gradcamUrl = null;

#   // Re-enable analyse button in case it was disabled
#   const btn = document.getElementById("analyseBtn");
#   if (btn) { btn.disabled = false; btn.textContent = "🔍 Analyse This Leaf"; }

#   // Re-show gradcam offer button in case it was hidden
#   const gcBtn = document.getElementById("gradcamOfferBtn");
#   if (gcBtn) gcBtn.style.display = "";

#   const reader = new FileReader();
#   reader.onload = e => {
#     document.getElementById("previewImg").src = e.target.result;
#     document.getElementById("previewLabel").textContent = file.name;
#     show("msgPreview");
#     setTimeout(() => { show("msgAnalysePrompt"); }, 400);
#     setTimeout(() => { show("msgAnalyseBtn"); scrollBottom(); }, 800);
#   };
#   reader.readAsDataURL(file);
# }

# // ── Run prediction ──
# function runPredict() {
#   const input = document.getElementById("imageInput");
#   if (!input.files.length) return;

#   hide("msgAnalyseBtn");
#   show("msgTyping");
#   scrollBottom();

#   const fd = new FormData();
#   fd.append("image", input.files[0]);

#   fetch("/predict", { method:"POST", body:fd })
#     .then(r => r.json())
#     .then(data => {
#       hide("msgTyping");

#       if (data.error) {
#         showErrorBubble("Hmm, something went wrong: " + data.error);
#         return;
#       }

#       gradcamUrl = data.gradcam;

#       // Format disease name nicely
#       const niceName = data.prediction.replace(/___/g, " — ").replace(/_/g, " ");
#       document.getElementById("diseaseName").textContent = niceName;
#       document.getElementById("resultEmoji").textContent = getResultEmoji(niceName);

#       setTimeout(() => { show("msgResult"); scrollBottom(); }, 200);
#       setTimeout(() => { show("msgConfirm"); scrollBottom(); }, 900);
      
#       setTimeout(() => { show("msgTryAgain"); scrollBottom(); }, 2300);
#     })
#     .catch(() => {
#       hide("msgTyping");
#       showErrorBubble("Prediction failed — is Flask running? 😅");
#     });
# }

# // ── Show gradcam ──
# function showGradcam() {
#   if (!gradcamUrl) return;
#   document.getElementById("gradcamImg").src = gradcamUrl;

#   // hide the offer bubble's button so it can't be clicked again
#   const ob = document.getElementById("gradcamOfferBtn"); if(ob) ob.style.display="none";

#   setTimeout(() => { show("msgGradcam"); scrollBottom(); }, 200);
# }

# // ── Helpers ──
# function show(id) {
#   const el = document.getElementById(id);
#   if (el) el.classList.remove("hidden");
# }
# function hide(id) {
#   const el = document.getElementById(id);
#   if (el) el.classList.add("hidden");
# }
# function scrollBottom() {
#   setTimeout(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }), 100);
# }
# function showErrorBubble(msg) {
#   const div = document.createElement("div");
#   div.className = "bubble-wrap";
#   div.innerHTML = `<div class="avatar">🤖</div><div class="bubble bot" style="color:#ff6b6b">${msg}</div>`;
#   document.getElementById("chat").appendChild(div);
#   scrollBottom();
# }

# // ── Drag & drop on upload zone ──
# const zone = document.getElementById("uploadZone");
# zone.addEventListener("dragover", e => { e.preventDefault(); zone.style.borderColor="var(--green)"; });
# zone.addEventListener("dragleave", () => { zone.style.borderColor=""; });
# zone.addEventListener("drop", e => {
#   e.preventDefault(); zone.style.borderColor="";
#   if (e.dataTransfer.files.length) {
#     document.getElementById("imageInput").files = e.dataTransfer.files;
#     onImagePicked();
#   }
# });
# </script>
# </body>
# </html>















# edge  case handling 
# import tensorflow as tf
# import numpy as np
# import cv2
# import os

# # ✅ predict_xai.py is in src/ → go one level up to reach project root
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # ------------------------------
# # LOAD MODEL
# # ✅ models are in src/models/
# # ------------------------------
# model = tf.keras.models.load_model(
#     os.path.join(ROOT_DIR, "src", "models", "best_crop_disease_model.keras")
# )
# print("✔ Model loaded")

# # ------------------------------
# # CLASS NAMES
# # ✅ dataset is now in datasets/train_dataset/
# # ------------------------------
# dataset_dir = os.path.join(ROOT_DIR, "datasets", "train_dataset")

# class_names = sorted([
#     d for d in os.listdir(dataset_dir)
#     if os.path.isdir(os.path.join(dataset_dir, d))
# ])
# print("✔ Classes:", class_names)

# # ------------------------------
# # LAST CONV LAYER
# # ------------------------------
# last_conv_name = "block_16_project"
# last_conv_layer = model.get_layer(last_conv_name)

# # ------------------------------
# # GRAD CAM
# # ------------------------------
# def generate_gradcam(image_path):

#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

#     grad_model = tf.keras.Model(
#         inputs=model.input,
#         outputs=[last_conv_layer.output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_out, predictions = grad_model(img_array)
#         pred_index = tf.argmax(predictions[0])
#         loss = predictions[:, pred_index]

#     grads = tape.gradient(loss, conv_out)[0]
#     conv_out = conv_out[0]

#     weights = tf.reduce_mean(grads, axis=(0, 1))
#     heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)

#     for i in range(weights.shape[0]):
#         heatmap += weights[i] * conv_out[:, :, i]

#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= (heatmap.max() + 1e-9)

#     original = cv2.imread(image_path)
#     heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
#     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#     result = cv2.addWeighted(original, 0.3, heatmap, 0.7, 0)

#     return int(pred_index), result


# # ------------------------------
# # PREDICT FUNCTION
# # ------------------------------
# def predict_for_web(image_path):
#     idx, grad_img = generate_gradcam(image_path)
#     predicted_class = class_names[idx]
#     return predicted_class, grad_img