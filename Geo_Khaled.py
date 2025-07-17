# ╭──────────────────────────────────────────────────────────────────────────╮
#   Streamlit | Sentinel-2 Water-Quality Dashboard (Basemaps + BloomRamp)    │
# ╰──────────────────────────────────────────────────────────────────────────╯
import io, datetime
import numpy as np
import plotly.express as px
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import cmocean
from sentinelhub import (
    SHConfig, SentinelHubRequest, MimeType,
    CRS, BBox, DataCollection, bbox_to_dimensions, SentinelHubCatalog
)
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.font_manager as fm

# ✅ تعديل: إضافة مكتبات للتعامل مع .env
import os  # مكتبة لإدارة المتغيرات البيئية
from dotenv import load_dotenv  # مكتبة لتحميل ملف .env
load_dotenv()  # ✅ تعديل: تحميل المتغيرات من ملف .env



def ar(text: str) -> str:
    """يعيد النص العربي مشكلاً ومرتباً RTL ليقبله matplotlib."""
    return get_display(arabic_reshaper.reshape(text))

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ======== دالة مساعدة لإعادة التشغيل ========
def rerun_app():
    """دالة مساعدة لإعادة تشغيل التطبيق مع دعم الإصدارات المختلفة"""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        # حل بديل باستخدام JavaScript
        js = """
        <script>
            window.location.reload();
        </script>
        """
        st.components.v1.html(js)
    raise st.StopException

# ======== تهيئة Sentinel Hub ========
try:
    config = SHConfig()

    # ✅ تعديل: استخدام متغيرات البيئة بدلًا من القيم المكتوبة مباشرة
    config.instance_id = os.getenv("INSTANCE_ID")
    config.sh_client_id = os.getenv("SH_CLIENT_ID")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

    if not all([config.instance_id, config.sh_client_id, config.sh_client_secret]):
        st.error("❌ بيانات اعتماد Sentinel Hub غير مكتملة!")
        st.stop()

except Exception as e:
    st.error(f"❌ خطأ في تهيئة الإعدادات: {str(e)}")
    st.stop()

# ─────────────────────────ـ تطبيق CSS لتنسيق النصوص العربية وتصميم الشريط الجانبي ـ──────────────────────
st.markdown("""
<style>

/* 1. اجعل الـ Header الافتراضي لـ Streamlit شفافًا */
[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0.0) !important; /* شفافية كاملة */
    /* يمكنك استخدام قيمة أقل من 1 لجعلها شبه شفافة، مثلاً:
       background-color: rgba(255, 255, 255, 0.2) !important; */
    border-bottom: none !important; /* إزالة أي حدود سفلية قد تظهر */
    box-shadow: none !important; /* إزالة أي ظل قد يظهر */
}

/* تنسيقات عامة للنصوص */
/* 1) اجعل كل الصفحة RTL تلقائيًّا (بدون !important) */
body {
    direction: rtl;
    text-align: right;
}

/* تنسيقات خاصة للعناوين */
h1, h2, h3 {
    text-align: center !important;
    font-weight: bold !important;
    color: #2c3e50 !important;
}

/* تنسيقات للشريط الجانبي (على اليسار) */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, rgba(255,228,225,0.9) 0%, rgba(255,248,220,0.9) 100%) !important;
    border-radius: 15px 0 0 15px;
    padding: 20px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.5);
    margin: 10px;
    right: 0 !important; /* التعديل هنا */
    left: auto !important; /* التعديل هنا */
    top: 0 !important;
    height: 100vh;
    overflow-y: auto;
    width: 320px !important;
    transition: all 0.3s ease; /* إضافة تحريك سلس */
}

/* إخفاء الشريط الجانبي عند الانكماش */
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: translateX(100%);
}

/* تنسيقات لأزرار الستريمليت */
.stButton>button {
    font-family: 'Arial', 'Tahoma', sans-serif !important;
    text-align: center !important;
    width: 100%;
    background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px !important;
    margin-top: 10px;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

/* تنسيقات لصناديق الاختيار والاختيار المنبثق */
.stCheckbox>label, .stSelectbox>label, .stNumberInput>label,
.stSlider>label, .stDateInput>label {
    direction: rtl !important;
    text-align: right !important;
    color: #6a5acd !important;
    font-weight: bold !important;
}

/* تنسيقات للتحذيرات والأخطاء */
.stAlert {
    direction: rtl !important;
    text-align: right !important;
    background-color: rgba(255, 228, 225, 0.7) !important;
    border-left: 4px solid #ff6b6b !important;
}

/* تنسيقات خاصة للقوائم */
ul {
    padding-right: 20px !important;
    direction: rtl !important;
}

/* تنسيقات للخرائط */
.map-container {
    width: calc(100% - 1rem) !important;
    margin-right: 0 !important;
    margin-left: auto !important;
    direction: ltr !important;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

/* تنسيقات للبطاقات */
.stExpander {
    background-color: rgba(255, 250, 240, 0.8) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 218, 185, 0.5) !important;
}

/* تنسيقات للألوان الرومانسية */
:root {
    --primary-color: #ff9a9e;
    --secondary-color: #fad0c4;
    --accent-color: #a18cd1;
    --text-color: #5a5a5a;
}

/* تنسيقات نافذة البداية */
.welcome-container {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    background: url('https://raw.githubusercontent.com/GisDune/Khaled/refs/heads/main/12.webp') center/cover no-repeat !important;
    background-attachment: fixed;        /* تأثير Parallax خفيف على الحواسيب */
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important; /* توسيط عمودي */
    align-items: center !important; /* توسيط أفقي */
    text-align: center !important; /* توسيط النص داخل الحاوية */
    box-sizing: border-box; /* التأكد من أن التبطين والحواف مشمولة في الحجم الكلي */

}
.welcome-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;           /* رفع المحتوى إلى أعلى الحاوية */
    align-items: center;
    padding: 4vh 20px 20px !important;     /* 4vh من الأعلى + 20px يمين/يسار/أسفل */
    margin: 0 !important;
    box-sizing: border-box;
    width: 100%;                           /* يشغل العرض بالكامل */

}

.st-emotion-cache-1kyxreq {
    padding: 0 !important;
}

/* تنسيقات الزر العام في التطبيق */
.stButton>button {
    font-size: 1.2rem !important;
    padding: 12px 24px !important;
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px 60px !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    margin: 0 auto !important;
    display: block !important;
    width: auto !important;
    max-width: 100vh !important;
}

/* --- welcome-title --- */
.welcome-title{
    font-size: 3.5rem !important;
    color: #000 !important;
    padding: 15px 30px !important;
    border-radius: 10px !important;
    text-align: center !important;
    margin: 0 !important;
    text-shadow: 0 2px 4px rgba(255,215,0,0.35);
    max-width: 95% !important;   /* كان 80% */
    width: 95% !important;       /* دعم إضافى لبعض المتصفحات */
}

/* --- welcome-subtitle --- */
.welcome-subtitle{
    font-size: 1.8rem !important;
    color: #000 !important;
    background: rgba(255,255,255,0.8) !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    text-align: center !important;
    margin-top: 20px !important;
    max-width: 95% !important;   /* توسعة العرض مثل العنوان */
    width: 95% !important;
}


.welcome-btn {
    font-size: 1.5rem;
    padding: 15px 40px;
    border-radius: 18px 66px;
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    margin-top: 20px;
}

.welcome-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}

.gradient-title{
    font-size:20px;
    font-weight:bold;
    text-align:center;
    direction:rtl;
    margin-top:0.3rem;
    margin-bottom:0.2rem;
}

/* زر البدء في نافذة الترحيب (التنسيقات الرئيسية) */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%) !important;
    width: 200px !important;
    height: 70px !important;
    font-size: 80px !important;
    font-weight: bold !important;
    border-radius: 35px !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
    transition: all 0.5s ease !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 0 !important;
    position: fixed !important; /* هذا يسمح لنا بتحديد موقعه بدقة */
    top: 65% !important; /* تم تعديله ليكون أقرب لأسفل الشاشة */
    left: 10% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 9999 !important;
    overflow: hidden !important;
    cursor: pointer !important;
    /* إزالة padding-left: 15px !important; لأنه كان يتسبب في إزاحة طفيفة */
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    width: 100px !important;
    height: 100px !important;
    border-radius: 50% !important;
    background: radial-gradient(
        circle at center,
        #4CAF50 0%,
        #388E3C 30%,
        #2E7D32 70%,
        #1B5E20 100%
    ) !important;
    transform: translate(-50%, -50%) scale(1.1) !important;
    box-shadow: 0 0 25px rgba(46, 125, 50, 0.6) !important;
    animation: rotateEarth 8s infinite linear !important;
}

@keyframes rotateEarth {
    from { background-position: 0 0; }
    to { background-position: 100% 0; }
}

div[data-testid="stButton"] > button[kind="primary"]:hover::after {
    content: "🌍";
    font-size: 40px;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}

/* تنسيق فريد لزر حساب المؤشر فقط */
div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"]
div[data-testid="stButton"] button[kind="primary"][data-testid="baseButton-secondary"] {
    all: unset !important;
    background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-size: 1.2rem !important;
    font-weight: bold !important;
    width: 100% !important;
    margin: 10px 0 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
    display: flex !important;
    justify-content: center !important;
    align-items: flex-end !important;
    margin-top: -20px !important; /* تحريك الزر لأعلى */
}

div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"]
div[data-testid="stButton"] button[kind="primary"][data-testid="baseButton-secondary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
    background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%) !important;
}

div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"]
div[data-testid="stButton"] button[kind="primary"][data-testid="baseButton-secondary"]:active {
    transform: translateY(1px) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
}



/* تنسيقات جديدة لتقليل المسافة بين الخريطة والزر */
.map-button-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 15px;
}

.map-button-group .stButton>button {
    margin-top: 5px !important;
    margin-bottom: 5px !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
}

/* تعديل حجم الخريطة */
.st-folium {
    margin-bottom: 0 !important;
}

/* تنسيقات متجاوبة للجوال والشاشات الصغيرة (كتلة واحدة مدمجة) */
@media (max-width: 768px) {
    /* تكييس الأعمدة */
    .st-emotion-cache-1cypcdb, .st-emotion-cache-1y4p8pa {
        flex-direction: column;
    }

    /* تعديل حجم الخريطة */
    .map-container {
        height: 300px !important;
    }
    
    /* تكييس الشريط الجانبي */
    [data-testid="stSidebar"] {
        width: 280px !important;
        border-radius: 15px 0 0 15px;
        transform: translateX(0);
        height: auto; /* السماح للشريط الجانبي بالتكيف مع المحتوى */
        padding: 10px !important; /* تقليل التبطين في الجوال */
        right: 0;
        left: auto !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] {
        transform: translateX(100%);
    }

    /* تعديلات العنوان على الجوال */
    .welcome-title {
        font-size: 1.4rem !important; /* حجم خط مناسب للجوال */
        padding: 10px !important; /* تقليل الهوامش الداخلية */
        line-height: 1.4 !important; /* تحسين ارتفاع السطور */
        text-shadow: 0 1px 2px rgba(255,215,0,0.35); /* ظل أخف */
        margin-top: 40vh !important; /* هامش علوي لتموضع أفضل */
        max-width: 100% !important; /* تحديد عرض أقصى للسماح بالالتفاف */
        word-wrap: break-word; /* كسر الكلمات الطويلة */
        white-space: normal; /* السماح بالتفاف النص بشكل طبيعي */
        box-sizing: border-box; /* تضمين التبطين والحواف */
        margin-bottom: 20px !important; /* مسافة بين العنوان والزر */
    }

    .welcome-subtitle {
        font-size: 1rem !important; /* تصغير حجم الخط الفرعي */
        padding: 8px 15px !important;
    }

    /* تعديلات زر البدء على الجوال */
    div[data-testid="stButton"] > button[kind="primary"] {
        font-size: 2rem !important; /* حجم خط أكبر للزر */
        width: 200px !important; /* عرض ثابت للزر */
        height: 60px !important; /* ارتفاع ثابت للزر */
        top: 80% !important; /* تغيير الموضع الرأسي للزر */
        left: 50% !important; /* توسيط أفقي */
        transform: translate(-50%, -50%) !important; /* توسيط دقيق */
        border-radius: 30px !important; /* زوايا مدورة */
        animation: pulse 2s infinite;  /* إضافة تأثير النبض */
        padding: 0 !important; /* إزالة التبطين الزائد */
    }

    /* إخفاء التأثيرات المعقدة على الجوال لزر البدء */
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        width: 200px !important; /* الحفاظ على نفس الحجم عند التحويم */
        height: 60px !important;
        border-radius: 30px !important;
        animation: pulse 2s infinite !important; /* استمرار النبض */
        transform: translate(-50%, -50%) !important; /* نفس المركز */
        box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important; /* ظل موحد */
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover::after {
        content: "" !important; /* إزالة أيقونة الأرض عند التحويم على الجوال */
    }
    
    /* تأثير النبض للزر على الجوال */
    @keyframes pulse {
        0% { transform: translate(-50%, -50%) scale(1); }
        50% { transform: translate(-50%, -50%) scale(1.05); }
        100% { transform: translate(-50%, -50%) scale(1); }
    }
}

/* شاشات متوسطة الحجم (أجهزة لوحية) */
@media (min-width: 769px) and (max-width: 1024px) {
    .welcome-title {
        font-size: 2.5rem !important;
    }
    .welcome-container{
        /* ❶ اجعل الصورة بالكامل داخل الإطار دون قصّ */
        background-size: contain !important;   /* بدلاً من cover */
        /* ❷ اجعلها تتكرر رأسيًّا إذا لازم الأمر حتى لا يظهر فراغ */
        background-repeat: no-repeat !important;
        background-position: top center !important;
    }
    #_______________________________________________________________________________________________
    
            
    #_________________________________________________________________________________________________
    .map-container {
        height: 400px !important;
    }
    
    div[data-testid="stButton"] > button[kind="primary"] {
        font-size: 3rem !important;
        width: 70% !important;
    }
}

/* تعديلات عامة للاستجابة */
.stPlotlyChart, .stImage {
    max-width: 100% !important;
    height: auto !important;
}

/* تكبير النصوص في العناصر الرئيسية */
h1, h2, h3 {
    font-size: calc(16px + 1vw) !important;
}

/* تكبير خطوط التسميات */
.stSelectbox label, .stSlider label, .stDateInput label {
    font-size: calc(12px + 0.5vw) !important;
}
            
            /* ========== الهواتف والأجهزة الصغيرة (≤ 768px) ========== */
@media (max-width: 768px){
    .welcome-container{
        /* 1) ألغِ الخلفيّة السابقة كلّياً ثم عرِّفها من جديد */
        background: url('https://raw.githubusercontent.com/GisDune/Khaled/refs/heads/main/b.jpg')
                    top center / contain              /* الحجم = contain */
                    no-repeat scroll !important;      /* لا قصّ ولا ثبات */

        /* 2) استبدل height:100vh بحدّ أدنى كى تسمح للتمرير إن احتجت */
        height: auto !important;
        min-height: 100vh !important;  /* تظلّ تغطّى الشاشة كاملة مع إمكانيّة التمدّد */
    }

    /* إزالة حوافّ Streamlit الافتراضية لتستفيد من عرض الهاتف بالكامل */
    section.main > div.block-container{
        padding: 0 !important;
        max-width: 100% !important;
    }
}
/* للشاشات المتوسطة والصغيرة */
@media (max-width: 772px){
.welcome-container {
    background:
        linear-gradient(
            to bottom,
            transparent 0%,
            transparent 10%,
            rgba(230,249,255,0.05) 12%,
            rgba(230,249,255,0.15) 20%,
            rgba(215,246,236,0.3) 35%,
            rgba(180,235,180,0.5) 55%,
            rgba(168,227,144,0.7) 75%,
            rgba(168,227,144,0.85) 90%,
            rgba(168,227,144,0.95) 100%
        ),
        url('https://raw.githubusercontent.com/GisDune/Khaled/refs/heads/main/b.jpg')
        top center / contain no-repeat scroll !important;

        min-height: 100vh !important;
        height: auto !important;
    }

    section.main > div.block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
}

/* للشاشات الصغيرة جداً مثل 322px */
@media (max-width: 340px){
.welcome-container {
    background:
        linear-gradient(
            to bottom,
            transparent 0%,
            transparent 1%,                   /* ⬅︎ نُقدّم التدرج قليلاً */
           
            rgba(180,235,180,0.5) 60%,
            rgba(168,227,144,0.75) 75%,
            rgba(168,227,144,0.9) 90%,
            rgba(168,227,144,1) 100%
        ),
        url('https://raw.githubusercontent.com/GisDune/Khaled/refs/heads/main/b.jpg')
        top center / contain no-repeat scroll !important;
    }
}


/* —————— إجبار السلايدر على LTR بشكل فعّال —————— */
/* ========== 1) اجعل الـ slider نفسه LTR بالكامل ========== */
    [data-testid="stSlider"] {
    direction: ltr !important;
    unicode-bidi: isolate-override !important;
    text-align: left !important;
}

/* ========== 2) عزل BaseWeb track/thumb داخليًا ========== */
[data-testid="stSlider"] div[data-baseweb="slider"] {
    direction: ltr !important;
    unicode-bidi: isolate-override !important;
    position: relative !important;
}

/* ========== 3) Thumb (الدائرة) والرقم فوقها ========== */
[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    direction: ltr !important;
    unicode-bidi: isolate-override !important;
    position: relative !important;
}
/* تنسيق مخصص لإظهار الدائرة الصغيرة وقيمة المؤشر */
div[data-baseweb="slider"] {
    position: relative;
    height: 32px;  /* هذا هو ما يسمح بظهور الدائرة الصغيرة فوق الشريط */
}





#_________________________________________________________________________________________________________________________

            
</style>
""", unsafe_allow_html=True)
# ──────────────────────── إعدادات الجلسة ───────────────────────
for k, v in [("img", None), ("label", ""), ("mdwi", None),
                ("bbox", None), ("size", None), ("scene_date", ""),
                ("show_welcome", True), ("show_main_app", False),
                ("show_exit_message", False)]:
    st.session_state.setdefault(k, v)

# ─────────────────────────── إعداد الصفحة ───────────────────────────
st.set_page_config(
    layout="wide",
    page_title="برنامج تحليل جودة المياه والغطاء النباتي",
    page_icon="🌊"
)

# ───────────────────────── BloomRamp colormap (Blue-→-Red) ─────────────────
bloom_cmap = LinearSegmentedColormap.from_list(
    "BloomRamp",
    ["#0020a5", "#01b3ff", "#ffff5e", "#ff9b00", "#c10000"],
    N=256
)

# ───────────────────────────── نافذة البداية ───────────────────────────────
def show_welcome_page():
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-content">
                <h1 class="welcome-title">
                        مرحبًا بك انت الان علي كوكب الارض والطبيعة بين يديك كما لم تراها من قبل حيث تلتقي علوم الاستشعار مع الطبيعة لتكشف لك اسرار البيئة من حولك
                </h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    clicked = st.button("بدء التطبيق", key="start_app", type="primary")
    
    if clicked:
        st.session_state.show_welcome = False
        st.session_state.show_main_app = True
        rerun_app()

# ───────────────────────────── رسالة الخروج ───────────────────────────────
def show_exit_message():
    st.markdown("""
    <div class="welcome-container">
        <h1 class="welcome-title">شكرًا لاستخدامك برنامجنا</h1>
        <h2 class="welcome-subtitle">تم الخروج من البرنامج بنجاح. نتمنى لك يومًا سعيدًا!</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("العودة إلى البداية", key="back_to_start", use_container_width=True):
        st.session_state.show_exit_message = False
        st.session_state.show_welcome = True
        rerun_app()

# ───────────────────────────── زر الخروج ───────────────────────────────
def show_exit_button():
    if st.sidebar.button("🚪 الخروج من البرنامج", use_container_width=True):
        st.session_state.show_main_app = False
        st.session_state.show_exit_message = True
        rerun_app()

# ───────────────────────────── التحكم في التدفق الرئيسي ───────────────────────────────
if st.session_state.get("show_exit_message", False):
    show_exit_message()
    st.stop()

if st.session_state.get("show_welcome", True):
    show_welcome_page()
    st.stop()

# ───────────────────────────── عناصر التحكم الجانبية ────────────────────────────
with st.sidebar:
    st.header("🗺️ إعدادات العرض")

    basemaps = {
        "خريطة Esri العالمية": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "OpenStreetMap":       "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "خريطة Stamen التضاريسية": "https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{y}/{x}.jpg"
    }

    basemap_url   = st.selectbox("خريطة الأساس", list(basemaps.keys()))
    basemap_tiles = basemaps[basemap_url]

    palette_options = ["haline", "viridis", "plasma", "RdYlGn_r",
                        "BloomRamp", "thermal", "algae"]

    palette_name = st.selectbox("لوحة الألوان", palette_options, index=0)

    auto_stretch = st.checkbox("قصّ تلقائي (P2–P98)", True)
    min_thr = st.number_input("القص الأدنى", value=-0.05, step=0.01, format="%.4f")
    max_thr = st.number_input("القص الأقصى", value=0.05,  step=0.01, format="%.4f")
    gamma = st.sidebar.slider("Gamma", 0.2, 3.0, 1.0, 0.1)

    
   # شرح معدل ليتناسب مع التصميم الجديد
    st.caption("""
    **تفسير القيم:**
    - **أقصى اليسار (3.00):** تظليل الألوان
    - **الوسط (1.0):** متوازن (افتراضي)
    - **أقصى اليمين (0.20):** تفتيح الألوان
    """)
    
    
    apply_mask = st.checkbox("🚿 إظهار المياه فقط (MDWI)", value=False, key="mask_toggle")
    log_chl    = st.checkbox("📈 تحويل لوغاريتمي لـ Chl_a", False)

   
    # ─── محدد نطاق التاريخ ──────────────────────────────
    st.markdown("📅 **اختر النطاق الزمني**")

    start_date = st.date_input(
        "تاريخ البداية:",
        value=datetime.date(2024, 6, 1),
        min_value=datetime.date(2015, 6, 23),
        max_value=datetime.date.today(),
        key="start_date_picker"
    )

    end_date = st.date_input(
        "تاريخ النهاية:",
        value=datetime.date(2024, 6, 25),
        min_value=start_date,
        max_value=datetime.date.today(),
        key="end_date_picker"
    )

    if start_date > end_date:
        st.error("⚠️ تاريخ النهاية يجب أن يكون بعد تاريخ البداية")
        st.stop()

    time_interval = (str(start_date), str(end_date))

    # زر الخروج داخل الشريط الجانبي
    show_exit_button()
# ← هنا ينتهى الـ with تلقائيًّا ـــــــــــــــــــــــــــــــــــــــ

# عناصر الصفحة الرئيسة (خارج الشريط)
st.title("منصة تحليل ومراقبة جودة المياه والغطاء النباتي بدقة مكانية 10 م 🌍")
st.markdown("---")


# قائمة خيارات المؤشرات (محدثة مع إضافة OSI)
indicator_keys = [
    "FAI (VB-FAI)", "MCI", "NDVI", "MDWI",
    "Chl_a (mg/m³)", "Cyanobacteria (10³ cells/ml)",
    "Turbidity (NTU)", "CDOM (mg/l)", "DOC (mg/l)", "Color (Pt-Co)",
    "OSI (Oil Spill Index)"  # مؤشر الانسكاب النفطي الجديد
]

indicator_display_names = {
    "FAI (VB-FAI)": "FAI (مؤشر الطحالب الطافية)",
    "MCI": "MCI (مؤشر الكلوروفيل الأقصى)",
    "NDVI": "NDVI (مؤشر الغطاء النباتي الطبيعي)",
    "MDWI": "MDWI (مؤشر المياه المعدل)",
    "Chl_a (mg/m³)": "Chl_a (كلوروفيل-أ بالمجم/م³)",
    "Cyanobacteria (10³ cells/ml)": "البكتيريا الزرقاء (آلاف خلية/مل)(Cyanobacteria)",
    "Turbidity (NTU)": "العكارة (NTU)",
    "CDOM (mg/l)": "المادة العضوية الملونة (ملجم/لتر)(CDOM)",
    "DOC (mg/l)": "الكربون العضوي المذاب (ملجم/لتر)(DOC)",
    "Color (Pt-Co)": "اللون (وحدات Pt-Co)",
    "OSI (Oil Spill Index)": "OSI (مؤشر الانسكاب النفطي)"  # اسم العرض الجديد
}

selected_indicator_display_name = st.selectbox(
    "اختر المؤشّر:",
    list(indicator_display_names.values())
)

indicator = next(key for key, value in indicator_display_names.items() if value == selected_indicator_display_name)

# ───────────────────────── Layout (left ↔ right) ───────────────────────────
left_col, right_col = st.columns([3, 1])

# ─────────────────────────── Folium map widget ─────────────────────────────
with left_col:
    # حاوية تجميع الخريطة والزر مع تقليل المسافة بينهما
    st.markdown('<div class="map-button-group">', unsafe_allow_html=True)
    
    m = folium.Map(location=[23, 30], zoom_start=6, tiles=None)
    folium.TileLayer(tiles=basemap_tiles, attr=basemap_url).add_to(m)
    Draw(draw_options={"rectangle": True},
            edit_options={"edit": False}).add_to(m)
    aoi = st_folium(m, height=450, width=None, use_container_width=True,
                    returned_objects=["all_drawings"])
    
    # ───────────────────────────── Calculation Button ───────────────────────────
    calculate_clicked = st.button(
        "🧮 احسب المؤشر",
        key="unique_calculate_button",
        type="primary",
        use_container_width=True,
        help="انقر لحساب المؤشر المحدد"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────── Evalscripts dict (محدث مع إضافة OSI) ────────────────────────────
evalscripts = {
    "FAI (VB-FAI)": (
        """//VERSION=3
function setup(){return{input:["B05","B06","B07","SCL"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    if(s.SCL==8||s.SCL==9||s.SCL==11) return [NaN];
    let bl=s.B05+(s.B07-s.B05)*((740-705)/(783-705));
    return [s.B06-bl];
}""", "FAI", "L2A"),

    "MCI": (
        """//VERSION=3
function setup(){return{input:["B04","B05","B06"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    let bl=s.B04+(s.B06-s.B04)*(705-665)/(740-665);
    return [s.B05-bl];
}""", "MCI", "L2A"),

    "NDVI": (
        """//VERSION=3
function setup(){return{input:["B04","B08"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    return [(s.B08-s.B04)/(s.B08+s.B04)];
}""", "NDVI", "L2A"),

    "MDWI": (
        """//VERSION=3
function setup(){return{input:["B03","B08"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    return [(s.B03-s.B08)/(s.B03+s.B08)];
}""", "MDWI", "L2A"),

    "Chl_a (mg/m³)": (
        """//VERSION=3
function setup(){return{input:["B03","B01"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    return [4.26*Math.pow(s.B03/s.B01,3.94)];
}""", "Chl_a", "L2A"),

    "Cyanobacteria (10³ cells/ml)": (
        """//VERSION=3
function setup(){return{input:["B03","B04","B02"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    return [115530.31*Math.pow((s.B03*s.B04)/s.B02,2.38)];
}""", "Cya", "L2A"),

    "Turbidity (NTU)": (
        """//VERSION=3
function setup(){return{input:["B03","B01"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    return [8.93*(s.B03/s.B01)-6.39];
}""", "Turb", "L2A"),

    "CDOM (mg/l)": (
        """//VERSION=3
function setup(){return{input:["B03","B04"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    return [537*Math.exp(-2.93*s.B03/s.B04)];
}""", "CDOM", "L1C"),

    "DOC (mg/l)": (
        """//VERSION=3
function setup(){return{input:["B03","B04"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    return [432*Math.exp(-2.24*s.B03/s.B04)];
}""", "DOC", "L1C"),

    "Color (Pt-Co)": (
        """//VERSION=3
function setup(){return{input:["B03","B04"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    return [25366*Math.exp(-4.53*s.B03/s.B04)];
}""", "Color", "L1C"),
    
    "OSI (Oil Spill Index)": (  # إضافة مؤشر الانسكاب النفطي
        """//VERSION=3
function setup(){return{input:["B02","B03","B04"],
                            output:{bands:1,sampleType:"FLOAT32"}};}
function evaluatePixel(s){
    return [(s.B03 + s.B04) / s.B02];
}""", "OSI", "L1C")
}

# ───────────────────────── Default ranges & descriptions (محدث مع إضافة OSI) ───────────────────
default_ranges = {
    "Chl_a": (0.0, 50.0),
    "Cya": (0.0, 100.0),
    "Turb": (0.0, 25.0),
    "CDOM": (0.0, 7.0),
    "DOC": (0.0, 50.0),
    "Color": (0.0, 60.0),
    "FAI": (-0.02, 0.15),
    "MCI": (-0.05, 0.25),
    "NDVI": (-0.5, 0.6),
    "OSI": (0.0, 0.5)  # نطاق مؤشر الانسكاب النفطي
}

descriptions = {
    "FAI": """
**مؤشر الطحالب الطافية (FAI)**
* **التعريف:** يقيس انحراف الانعكاسية بالقرب من 740 نانومتر.
* **تفسير القيم:**
    * **-0.05 إلى 0.00:** مياه صافية
    * **0.00 - 0.05:** تركيز منخفض
    * **> 0.05 - 0.10:** تركيز متوسط
    * **> 0.10:** تركيز عالي
* **المدى المقترح:** -0.02 إلى 0.15
""",

    "MCI": """
**مؤشر الكلوروفيل الأقصى (MCI)**
* **التعريف:** يقيس تركيز الكلوروفيل في الماء.
* **تفسير القيم:**
    * **-0.05 إلى 0.00:** مياه صافية
    * **0.00 - 0.05:** كلوروفيل منخفض
    * **> 0.05 - 0.10:** كلوروفيل متوسط
    * **> 0.10:** كلوروفيل عالي
* **المدى المقترح:** -0.05 إلى 0.25
""",

    "NDVI": """
**مؤشر الغطاء النباتي (NDVI)**
* ** التعريف:** يستخدم للتمييز بين الماء والنباتات والكشف عن النبات الصحي اعتمادا علي نسبة محتوي الكلوروفيل المستويات العالية مؤشر جيد علي صحة النبات والمحتوي الرطوبي
* **تفسير القيم:**
    * **-1.0 إلى 0.00:** مياه صافية
    * **0.00 - 0.10:** نباتات متناثرة
    * **> 0.10 - 0.20:** غطاء نباتي متوسط
    * **> 0.20 - 0.50:** غطاء نباتي كثيف
    * **> 0.50:** غطاء نباتي كثيف جداً
* **المدى المقترح:** -0.5 إلى 0.6
""",

    "MDWI": """
**مؤشر المياه المعدل (MDWI)**
* **التعريف:** يستخدم للتمييز بين الماء واليابسة.
* **تفسير القيم:**
    * **< 0.0:** يابسة
    * **> 0.0:** مياه
    * **0.2 - 0.7:** مياه صافية
    * **> 0.7:** مياه عميقة
* **المدى المقترح:** -0.5 إلى 0.7
""",

    "Chl_a": """
**الكلوروفيل-أ (Chl_a)**
* **التعريف:** تركيز الكلوروفيل-أ بالمجم/م³.
* **تفسير القيم:**
    * **< 5:** مياه نظيفة
    * **5 - 10:** تغذية متوسطة
    * **10 - 25:** بداية ازدهار
    * **> 25 - 50:** ازدهار كثيف
    * **> 50:** ازدهار خطير
* **المدى المقترح:** 0.0 إلى 50.0
""",

    "Cya": """
**البكتيريا الزرقاء (Cyanobacteria)**
* **التعريف:** تركيز الخلايا (آلاف خلية/مل).
* **تفسير القيم:**
    * **0 - 10:** منخفض
    * **> 10 - 20:** مراقبة
    * **> 20 - 100:** تحذير صحي
    * **> 100:** خطر مباشر
* **المدى المقترح:** 0.0 إلى 100.0
""",

    "Turb": """
**العكارة (Turbidity)**
* **التعريف:** قياس تشتت الضوء (NTU).
* **تفسير القيم:**
    * **< 5:** صافية
    * **5 - 10:** خفيفة
    * **10 - 25:** متوسطة
    * **> 25 - 50:** عالية
    * **> 50:** تلوث شديد
* **المدى المقترح:** 0.0 إلى 25.0
""",

    "CDOM": """
**المادة العضوية الملونة (CDOM)**
* **التعريف:** تركيز المواد العضوية (ملجم/لتر).
* **تفسير القيم:**
    * **0.0 - 1.0:** منخفض
    * **> 1.0 - 3.0:** معتدل
    * **> 3.0:** مرتفع
* **المدى المقترح:** 0.0 إلى 7.0
""",

    "DOC": """
**الكربون العضوي المذاب (DOC)**
* **التعريف:** تركيز الكربون (ملجم/لتر).
* **تفسير القيم:**
    * **0.0 - 5.0:** منخفض
    * **5 - 10:** معتدل
    * **10 - 20:** مرتفع
    * **> 20:** تلوث شديد
* **المدى المقترح:** 0.0 إلى 50.0
""",

    "Color": """
**لون المياه (Pt-Co)**
* **التعريف:** قياس اللون الظاهر.
* **تفسير القيم:**
    * **0 - 15:** صافية
    * **15 - 40:** ملونة
    * **> 40:** داكنة
* **المدى المقترح:** 0.0 إلى 60.0
""",
    
    "OSI": """  # وصف مؤشر الانسكاب النفطي
**مؤشر الانسكاب النفطي (OSI)**
* **التعريف:** يقيس وجود انسكابات نفطية على سطح الماء باستخدام النطاقات المرئية (الأخضر، الأحمر، الأحمر الحدودي).
* **تفسير القيم:**
    * **0.0 - 0.1:** مياه نظيفة
    * **0.1 - 0.2:** مشتبه به (تلوث خفيف)
    * **0.2 - 0.3:** انسكاب نفطي محتمل
    * **> 0.3:** انسكاب نفطي مؤكد
* **المعادلة:** (B03 + B04) / B02
* **المدى المقترح:** 0.0 إلى 0.5
* **المراجع العلمية:**
    - Rajendran et al. (2021) - Oil spill detection using Sentinel-2
    - Rajendran et al. (2021) - Mapping oil spills in the Indian Ocean
"""
}

# قاموس لتخزين النطاقات الرقمية (محدث مع إضافة OSI)
indicator_numerical_points = {}
for key, (min_val, max_val) in default_ranges.items():
    mid_val = (min_val + max_val) / 2
    if key in ["FAI", "MCI", "NDVI", "MDWI", "OSI"]:  # إضافة OSI
        indicator_numerical_points[key] = {
            "min": f"{min_val:.2f}",
            "mid": f"{mid_val:.2f}",
            "max": f"{max_val:.2f}"
        }
    else:
        indicator_numerical_points[key] = {
            "min": f"{min_val:.1f}",
            "mid": f"{mid_val:.1f}",
            "max": f"{max_val:.1f}"
        }

# المؤشرات التي تحتاج قناع مياه (محدث مع إضافة OSI)
water_masked_indicators = ["FAI", "MCI", "Cya", "Turb", "Chl_a", "CDOM", "DOC", "Color", "OSI"]

# ───────────────────────────── Calculation ─────────────────────────────────
if calculate_clicked:
    drawings = aoi.get("all_drawings", [])
    if not drawings:
        st.warning("✋ الرجاء رسم منطقة الاهتمام أولاً")
        st.stop()

    coords = drawings[-1]["geometry"]["coordinates"][0]
    lons, lats = [p[0] for p in coords], [p[1] for p in coords]
    bbox = BBox([min(lons), min(lats), max(lons), max(lats)], CRS.WGS84)
    size = bbox_to_dimensions(bbox, 10)
    if max(size) > 2500:
        r = 2500 / max(size)
        size = (int(size[0] * r), int(size[1] * r))

    ev, label, tier = evalscripts[indicator]
    st.session_state.update({"label": label, "bbox": bbox, "size": size})

    dc = DataCollection.SENTINEL2_L1C if tier == "L1C" else DataCollection.SENTINEL2_L2A

    # ─── تحديد أحدث تاريخ متاح ───
    cat = SentinelHubCatalog(config=config)
    try:
        search_iter = cat.search(
            dc,
            bbox=bbox,
            time=time_interval,
            fields={"include": ["properties.datetime"], "exclude": ["links", "assets"]}
        )
        dates = [item["properties"]["datetime"][:10] for item in search_iter]
    except Exception as e:
        st.error(f"❌ تعذّر البحث عن التواريخ المتاحة: {e}")
        st.stop()

    if not dates:
        st.warning("⚠️ لا توجد مرئيات متاحة في هذا النطاق الزمني. جرّب تواريخ أخرى.")
        st.stop()

    selected_date = max(dates)
    st.session_state["scene_date"] = selected_date
    time_interval_single = (selected_date, selected_date)

    req = SentinelHubRequest(
        evalscript=ev,
        input_data=[SentinelHubRequest.input_data(
            data_collection=dc,
            time_interval=time_interval_single,
            mosaicking_order="mostRecent"
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox, size=size, config=config
    )
    try:
        st.session_state["img"] = req.get_data()[0]
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

    # ─── تحميل قناع المياه إذا كان المؤشر يتطلبه ───
    if label in water_masked_indicators:
        mask_req = SentinelHubRequest(
            evalscript=evalscripts["MDWI"][0],
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval_single,
                mosaicking_order="mostRecent"
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox, size=size, config=config
        )
        try:
            st.session_state["mdwi"] = mask_req.get_data()[0]
        except Exception as e:
            st.warning(f"⚠️ تعذّر تحميل قناع المياه: {e}")

# ─────────────────────────── Display (left_col) ────────────────────────────
if st.session_state["img"] is not None:
    with left_col:
        img = st.session_state["img"].astype(np.float32)

        if st.session_state["label"] == "Chl_a" and log_chl:
            img = np.log1p(img)

        if apply_mask and st.session_state["label"] in water_masked_indicators \
                and st.session_state["mdwi"] is not None:
            mask = st.session_state["mdwi"].squeeze()
            img[mask <= 0] = np.nan

        real_min, real_max = np.nanmin(img), np.nanmax(img)
        st.sidebar.markdown(f"**min / max قبل القصّ:** {real_min:.3f} – {real_max:.3f}")

        if auto_stretch:
            p2, p98 = np.percentile(img[~np.isnan(img)], [2, 98])
            min_thr, max_thr = float(p2), float(p98)
        else:
            if (min_thr == -0.05 and max_thr == 0.05
                        and st.session_state["label"] in default_ranges):
                min_thr, max_thr = default_ranges[st.session_state["label"]]
        if max_thr - min_thr < 1e-6:
            max_thr += 1e-6

        # اختيار لوحة الألوان
        if hasattr(cmocean.cm, palette_name):
            cmap = getattr(cmocean.cm, palette_name)
        elif palette_name == "BloomRamp":
            cmap = bloom_cmap
        else:
            cmap = mpl.colormaps.get_cmap(palette_name)

        img_clip = np.clip(img, min_thr, max_thr)
        norm = (img_clip - min_thr) / (max_thr - min_thr)
        rgba = cmap(np.power(norm, gamma))
        rgb = (rgba[..., :3] * 255).astype(np.uint8)

        # عرض الصورة باستخدام plotly
        fig = px.imshow(rgb, origin="upper")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        scene_date = st.session_state.get("scene_date", "")
        st.sidebar.markdown(f"**📅 تاريخ المشهد:** {scene_date}")

        # تحسين عرض caption للصورة الرئيسية
        caption_text = (
            f"🖼️ مؤشر {indicator_display_names.get(indicator, st.session_state['label'])} "
            f"(تاريخ {scene_date})\nالمدى المعروض: {min_thr:.3f} – {max_thr:.3f}"
        )
        st.image(rgb, caption=caption_text, use_container_width=True)

        # ─── مفتاح التدرّج النصي (محدث مع إضافة OSI) ───
        legends = {
            "FAI": ["ضعيف", "متوسط", "مرتفع"],
            "MCI": ["منخفض", "متوسط", "مرتفع"],
            "NDVI": ["ضعيف", "متوسط", "كثيف"],
            "MDWI": ["يابسة", "مختلط", "مياه"],
            "Chl_a": ["منخفض", "متوسط", "مرتفع"],
            "Cya": ["منخفض", "متوسط", "مرتفع"],
            "Turb": ["منخفض", "متوسط", "مرتفع"],
            "CDOM": ["منخفض", "متوسط", "مرتفع"],
            "DOC": ["منخفض", "متوسط", "مرتفع"],
            "Color": ["فاتح", "متوسط", "غامق"],
            "OSI": ["نظيف", "مشتبه", "انسكاب"]  # تسميات OSI
        }
        labels_text = legends.get(st.session_state["label"], ["منخفض", "متوسط", "مرتفع"])
        labels_text = [ar(t) for t in labels_text]
        fig, ax = plt.subplots(figsize=(10, 1.5))
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(labels_text, fontsize=14)
        ax.set_yticks([])
        # ══════ هنا نضع tight_layout ══════
        plt.tight_layout(pad=3)  # بعد إعداد جميع العناصر وقبل حفظ الصورة

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.5, dpi=720)
        st.markdown(f"<p class='gradient-title'>🔎  التفسير النصي والرقمي للتدرج اللوني للانعكاسات الطيفية</p>",
            unsafe_allow_html=True)

        st.image(buf.getvalue(), use_container_width=True)
        plt.close(fig)

        # ─── مفتاح التدرّج الرقمي (3 قيم) ───
        if st.session_state["label"] in indicator_numerical_points:
            num_points = indicator_numerical_points[st.session_state["label"]]
            
            fig_num, ax_num = plt.subplots(figsize=(8, 0.5))
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax_num.imshow(gradient, aspect="auto", cmap=cmap)
            ax_num.set_xticks([0, 128, 255])
            ax_num.set_xticklabels(
                [num_points["min"], num_points["mid"], num_points["max"]],
                fontsize=12
            )
            ax_num.set_yticks([])
            ax_num.tick_params(axis='x', length=0)
            ax_num.set_frame_on(False)

            buf_num = io.BytesIO()
            fig_num.savefig(buf_num, format="png", bbox_inches="tight", pad_inches=0)
            st.image(buf_num.getvalue(), use_container_width=True)
            plt.close(fig_num)

# ───────────────────── شرح المؤشّر (right_col) ────────────────────────────
with right_col:
    if st.session_state.get("label"):
        with st.expander("📘 شرح المؤشّر", expanded=True):
            st.markdown(
                descriptions.get(
                    st.session_state["label"],
                    "لا يوجد وصف متوفر لهذا المؤشر"
                )
            )

