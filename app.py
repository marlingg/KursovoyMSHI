import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


FOLDER = Path(__file__).resolve().parent
LABELS = {
    "WALKING": "Ходьба",
    "WALKING_UPSTAIRS": "Піднімання сходами",
    "WALKING_DOWNSTAIRS": "Спускання сходами",
    "SITTING": "Сидіння",
    "STANDING": "Стояння",
    "LAYING": "Лежання",
}

st.set_page_config(page_title="Розпізнавання активності", page_icon="🚶",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {max-width: 1320px; padding-top: 2rem; padding-bottom: 2rem;}
.har-brand {font-size: 22px; font-weight: 750; letter-spacing: -.6px;}
.har-brand span {color: #8daeff;}
.har-kicker {font-size: 11px; letter-spacing: 2px; color: #a8bff4; font-weight: 700; margin-bottom: 12px;}
.har-hero {padding: 30px 34px; border: 1px solid #30405c; border-radius: 18px;
 background: linear-gradient(115deg, #152238, #101722); margin-bottom: 20px; color: #edf2fc;}
.har-hero h1 {font-size: clamp(27px, 3vw, 42px); line-height: 1.12; letter-spacing: -1.2px;
 color: #edf2fc; padding: 0; margin: 0 0 14px; max-width: 780px;}
.har-hero p {color: #b5c2d8; max-width: 700px; font-size: 15px; line-height: 1.65; margin: 0;}
.har-tag {display: inline-block; border: 1px solid #3b4e70; border-radius: 20px;
 font-size: 11px; padding: 5px 10px; margin: 19px 6px 0 0; color: #c4d4f6;}
.har-step {display:flex; align-items:center; gap:12px; padding: 9px 0;}
.har-step b {display:grid; place-items:center; width:29px; height:29px; border-radius:9px;
 background:#1f3050; color:#a9c4ff; font-size:12px; flex-shrink:0;}
.har-step span {font-size:14px; line-height:1.5;}
.har-empty {padding: 34px 22px; text-align:center; border:1px dashed #3b4b64;
 border-radius:14px; margin:10px 0; color:#b2bfd4; background:#101824;}
.har-empty strong {display:block; color:#edf2fc; font-size:18px; margin-bottom:9px;}
.har-empty p {font-size:14px; margin:0; line-height:1.65;}
.har-label {font-size:11px; letter-spacing:1.2px; opacity:.65; margin:20px 0 8px;}
.st-key-upload_panel, .st-key-help_panel, .st-key-results_panel {border-radius:16px;}
.st-key-recognize button {min-height:48px; font-weight:650; border-radius:10px;}
@media (max-width: 700px) {
 .har-hero {padding:24px 20px;} .block-container {padding-top:1rem;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(path, modified_time):
    # The timestamp invalidates the cached model when its file is replaced.
    return load_model(path, compile=False)


model_path = FOLDER / "model.h5"
info_path = FOLDER / "model_info.json"
info = None
settings_error = None
if info_path.exists():
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if (not isinstance(info.get("columns"), list) or len(info["columns"]) != 561
                or not all(isinstance(c, str) for c in info["columns"])
                or len(set(info["columns"])) != 561
                or not isinstance(info.get("classes"), list)
                or len(info["classes"]) != 6
                or set(info["classes"]) != set(LABELS)):
            raise ValueError("Очікуються 561 унікальна ознака та шість класів активності.")
    except Exception as error:
        info = None
        settings_error = str(error)
ready = model_path.is_file() and info is not None

with st.sidebar:
    st.markdown('<div class="har-brand">HAR<span> / </span>Аналіз руху</div>', unsafe_allow_html=True)
    st.caption("Розпізнавання активності людини")
    st.divider()
    st.markdown("**Робочий простір**")
    st.markdown("Аналіз CSV-файла")
    st.markdown('<div class="har-label">СТАН СИСТЕМИ</div>', unsafe_allow_html=True)
    if ready:
        st.info("Файли моделі підключено")
        st.caption("Модель завантажиться під час першого розпізнавання.")
    else:
        st.warning("Модель потребує налаштування")
    st.markdown('<div class="har-label">ВИДИ АКТИВНОСТІ</div>', unsafe_allow_html=True)
    for number, label in enumerate(LABELS.values(), 1):
        st.markdown(f"`0{number}`  {label}")
    st.divider()
    st.caption("Працює з готовими ознаками UCI HAR. Записи x/y/z зі смартфона потребують попереднього перетворення.")

st.markdown("""
<div class="har-hero">
 <div class="har-kicker">АНАЛІЗ ФІЗИЧНОЇ АКТИВНОСТІ</div>
 <h1>Від даних — до розпізнаного руху</h1>
 <p>Завантажте CSV із підготовленими ознаками. Модель визначить активність
 для кожного запису, а ви зможете переглянути та зберегти результати.</p>
 <span class="har-tag">6 видів активності</span>
 <span class="har-tag">Згорткова нейронна мережа</span>
 <span class="har-tag">Експорт у CSV</span>
</div>
""", unsafe_allow_html=True)

if settings_error:
    st.error(f"Не вдалося прочитати model_info.json: {settings_error}")
elif not ready:
    st.warning("Додайте model.h5 і model_info.json поруч із app.py. Обидва файли потрібно отримати з одного навчання моделі.")

upload_column, help_column = st.columns([1.65, 1], gap="large")
with upload_column:
    with st.container(border=True, key="upload_panel"):
        st.subheader("01 / Завантаження даних")
        st.caption("Оберіть файл для аналізу. Максимальний розмір — 30 МБ.")
        uploaded = st.file_uploader("Оберіть CSV-файл", type=["csv"], key="csv_upload",
                                    help="Наприклад, test.csv з навчального набору UCI HAR.")
        st.caption("561 числова ознака · заголовки колонок обов’язкові")
with help_column:
    with st.container(border=True, key="help_panel"):
        st.subheader("Як це працює")
        st.markdown("""
        <div class="har-step"><b>1</b><span>Завантажте файл і перевірте дані</span></div>
        <div class="har-step"><b>2</b><span>Натисніть «Розпізнати активність»</span></div>
        <div class="har-step"><b>3</b><span>Перегляньте прогноз і збережіть CSV</span></div>
        """, unsafe_allow_html=True)

with st.expander("Який CSV підходить для аналізу?"):
    st.write("Потрібні ті самі 561 ознака, що використовувалися під час навчання: один рядок заголовків і числові значення без пропусків. Порядок колонок може відрізнятися.")
    st.write("Колонки Activity та subject необов’язкові. Якщо Activity є, початкові мітки будуть показані поруч із прогнозом. Сирі записи Sensor Logger та phyphox ця версія не перетворює.")
    if info is not None:
        st.caption("Перші п’ять очікуваних колонок:")
        st.code(", ".join(info["columns"][:5]), language=None)

data = None
values = None
signature = None
if uploaded is not None:
    try:
        raw = uploaded.getvalue()
        if len(raw) > 30 * 1024 * 1024:
            raise ValueError("Файл перевищує 30 МБ. Завантажте меншу частину даних.")
        data = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        # Discard the previous result as soon as a different input is selected.
        if ready:
            signature = (hashlib.sha256(raw).hexdigest(), model_path.stat().st_mtime_ns,
                         info_path.stat().st_mtime_ns)
        saved = st.session_state.get("prediction")
        if saved and saved[0] != signature:
            st.session_state.pop("prediction", None)
        a, b, c = st.columns(3)
        a.metric("Записів у файлі", f"{len(data):,}".replace(",", " "), border=True)
        b.metric("Колонок", len(data.columns), border=True)
        c.metric("Початкові мітки", "Є" if "Activity" in data else "Немає", border=True)
        with st.expander("Переглянути вхідні дані · перші 20 записів", expanded=True):
            st.dataframe(data.head(20), hide_index=True, width="stretch")
        if ready:
            features = data.drop(columns=["Activity", "subject"], errors="ignore")
            missing = [c for c in info["columns"] if c not in features.columns]
            extra = [c for c in features.columns if c not in info["columns"]]
            if missing or extra:
                st.error("Формат не підходить: потрібні 561 ознака з навчального набору.")
                if missing:
                    st.write("Відсутні колонки (перші 5):", missing[:5])
                if extra:
                    st.write("Зайві колонки (перші 5):", extra[:5])
            elif data.empty:
                st.warning("У файлі є заголовки, але немає записів для аналізу.")
            else:
                values = features[info["columns"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
                if not np.isfinite(values).all():
                    values = None
                    st.error("У даних є пропуски або нескінченні значення. Виправте їх і завантажте файл повторно.")
                else:
                    st.success("Перевірку пройдено. Дані готові до розпізнавання.")
    except Exception as error:
        values = None
        st.session_state.pop("prediction", None)
        st.error(f"Не вдалося прочитати CSV: {error}")
else:
    st.session_state.pop("prediction", None)

can_predict = ready and values is not None
action, explanation = st.columns([1, 1.65], gap="large")
with action:
    predict = st.button("Розпізнати активність", type="primary", disabled=not can_predict,
                        width="stretch", key="recognize")
with explanation:
    st.caption("Модель обробить усі записи у файлі." if can_predict else
               "Кнопка стане доступною після завантаження сумісного CSV та підключення моделі.")

if predict and can_predict:
    st.session_state.pop("prediction", None)
    try:
        with st.spinner("Модель аналізує дані…"):
            model = get_model(str(model_path), model_path.stat().st_mtime_ns)
            if tuple(model.input_shape[1:]) != (561, 1) or model.output_shape[-1] != 6:
                raise ValueError("Форма входу чи виходу model.h5 не відповідає цій системі.")
            X = np.expand_dims(values, axis=2)
            probabilities = np.concatenate([model(X[i:i + 128], training=False).numpy()
                                             for i in range(0, len(X), 128)])
            if not np.isfinite(probabilities).all():
                raise ValueError("Модель повернула некоректні значення. Перевірте вхідні дані.")
            predictions = [info["classes"][i] for i in probabilities.argmax(axis=1)]
            result = pd.DataFrame({"Рядок": np.arange(1, len(data) + 1),
                                   "Прогноз": predictions,
                                   "Активність": [LABELS[name] for name in predictions],
                                   "Оцінка моделі": probabilities.max(axis=1)})
            if "Activity" in data:
                result.insert(1, "Activity з CSV", data["Activity"].to_numpy())
            st.session_state["prediction"] = (signature, result)
    except Exception as error:
        st.error(f"Не вдалося виконати розпізнавання: {error}")

st.divider()
with st.container(key="results_panel"):
    st.subheader("02 / Результати аналізу")
    saved = st.session_state.get("prediction")
    if can_predict and saved and saved[0] == signature:
        result = saved[1]
        counts = result["Активність"].value_counts()
        a, b, c = st.columns(3)
        a.metric("Оброблено записів", len(result), border=True)
        b.metric("Виявлено видів активності", len(counts), border=True)
        c.metric("Найчастіша активність", counts.index[0], border=True)
        table_column, chart_column = st.columns([1.5, 1], gap="large")
        with table_column:
            st.markdown("**Прогноз для кожного запису**")
            st.dataframe(result, hide_index=True, width="stretch",
                         column_config={"Оцінка моделі": st.column_config.NumberColumn(format="%.3f")})
        with chart_column:
            st.markdown("**Розподіл активностей**")
            chart = counts.rename_axis("Активність").reset_index(name="Кількість записів")
            st.bar_chart(chart, x="Активність", y="Кількість записів", horizontal=True, color="#6996ff", height=300)
        st.caption("Оцінка моделі — значення Softmax від 0 до 1. Вона не є гарантією правильності прогнозу.")
        st.download_button("Завантажити результати · CSV", result.to_csv(index=False).encode("utf-8-sig"),
                           file_name="result.csv", mime="text/csv", width="content")
    else:
        st.markdown("""<div class="har-empty"><strong>Тут з’явиться результат розпізнавання</strong>
        <p>Після аналізу ви побачите таблицю прогнозів, розподіл активностей<br>
        і кнопку завантаження результатів.</p></div>""", unsafe_allow_html=True)
st.caption("Розпізнавання активності людини · UCI HAR · CNN")
