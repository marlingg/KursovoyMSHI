import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# Уся програма інтерфейсу в одному файлі.
FOLDER = Path(__file__).resolve().parent

st.set_page_config(page_title='Розпізнавання активності', page_icon='🚶')
st.title('Розпізнавання активності людини')
st.write('Завантаж CSV → переглянь дані → натисни «Розпізнати».')
st.caption('Формат: 561 ознака з твого датасету Kaggle. Колонки Activity та subject можна залишити.')

uploaded = st.file_uploader('CSV-файл', type=['csv'])

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded, sep=None, engine='python')
        st.subheader('Вхідні дані')
        st.write(f'Рядків: {len(data)}, колонок: {len(data.columns)}')
        st.dataframe(data.head(20))

        model_path = FOLDER / 'model.h5'
        settings_path = FOLDER / 'model_info.json'

        if not model_path.exists() or not settings_path.exists():
            st.info('Спочатку виконай ноутбук test.ipynb. Він збереже model.h5 та model_info.json поруч із app.py.')
        else:
            info = json.loads(settings_path.read_text(encoding='utf-8'))
            columns = info['columns']
            classes = info['classes']
            features = data.drop(columns=['Activity', 'subject'], errors='ignore')
            missing = [c for c in columns if c not in features.columns]
            extra = [c for c in features.columns if c not in columns]

            if missing or extra:
                st.error('CSV не відповідає навчальним даним. Потрібні ті самі 561 колонка ознак, що у train.csv. Сирі x/y/z із телефона ця модель не приймає.')
                if missing:
                    st.write('Приклади відсутніх колонок:', missing[:5])
                if extra:
                    st.write('Зайві колонки:', extra[:5])
            elif len(data) == 0:
                st.warning('Файл не містить рядків даних.')
            else:
                # Той самий порядок колонок, що був під час навчання.
                features = features[columns].apply(pd.to_numeric, errors='raise')
                values = features.to_numpy(dtype=np.float32)
                if not np.isfinite(values).all():
                    st.error('У CSV є порожні значення, NaN або нескінченність.')
                else:
                    st.success('Формат CSV підходить до твоєї моделі.')
                    if st.button('Розпізнати', type='primary'):
                        with st.spinner('Розпізнавання…'):
                            model = load_model(model_path, compile=False)
                            if tuple(model.input_shape[1:]) != (len(columns), 1) or model.output_shape[-1] != len(classes):
                                raise ValueError('model.h5 не відповідає цьому ноутбуку та model_info.json.')
                            # Як у твоєму ноутбуці: (рядки, 561) → (рядки, 561, 1).
                            X = np.expand_dims(values, axis=2)
                            probabilities = np.concatenate([
                                model(X[i:i + 128], training=False).numpy()
                                for i in range(0, len(X), 128)
                            ])
                            predicted = probabilities.argmax(axis=1)
                            result = pd.DataFrame({
                                'Рядок': np.arange(1, len(data) + 1),
                                'Прогноз': [classes[i] for i in predicted],
                                'Оцінка моделі': probabilities.max(axis=1),
                            })
                            if 'Activity' in data:
                                result.insert(1, 'Activity з CSV', data['Activity'].to_numpy())
                        # У session_state тримаємо результат саме для поточного файла та моделі.
                        st.session_state['result'] = (uploaded.getvalue(), model_path.stat().st_mtime_ns,
                                                     settings_path.stat().st_mtime_ns, result)

                    saved = st.session_state.get('result')
                    if saved is not None and saved[:3] == (uploaded.getvalue(), model_path.stat().st_mtime_ns,
                                                          settings_path.stat().st_mtime_ns):
                        result = saved[3]
                        st.subheader('Результат')
                        st.dataframe(result)
                        st.caption('Оцінка моделі — softmax, не гарантована ймовірність правильності.')
                        st.bar_chart(result['Прогноз'].value_counts())
                        st.download_button('Завантажити результат CSV',
                                           result.to_csv(index=False).encode('utf-8-sig'),
                                           'result.csv', 'text/csv')
    except Exception as error:
        st.error(f'Не вдалося обробити файл: {error}')
