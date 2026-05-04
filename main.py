import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import os

# ==========================================
# 1. ЗАГРУЗКА И ГЛУБОКИЙ АНАЛИЗ (Lags & Rolling)
# ==========================================
def load_and_boost_ultra():
    print("--- 🚀 Подготовка данных (Lags & High Precision) ---")
    train = pd.read_csv('train.csv', low_memory=False)
    weather = pd.read_csv('weather.csv').drop_duplicates(subset=['Date'])
    holidays = pd.read_csv('holidays.csv').drop_duplicates(subset=['Date'])

    for df in [train, weather, holidays]:
        df['Date'] = pd.to_datetime(df['Date'])

    # Склейка
    data = train.merge(holidays[['Date', 'IsHoliday']], on='Date', how='left').fillna({'IsHoliday': 0})
    data = data.merge(weather[['Date', 'Temp', 'Rain']], on='Date', how='left').fillna({'Temp': 15, 'Rain': 0})

    # Жесткая чистка (удаляем праздники и воскресенья)
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data = data[(data['IsHoliday'] == 0) & (data['DayOfWeek'] != 6) & (data['Sales'] > 0)].copy()

    # Сортировка для лагов
    data = data.sort_values(['Store', 'Date'])

    # ПАМЯТЬ: Продажи 7 дней назад и скользящее среднее
    data['Sales_Lag_7'] = data.groupby('Store')['Sales'].shift(7)
    data['Rolling_7'] = data.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(window=7).mean())
    data['Promo_Tomorrow'] = data.groupby('Store')['Promo'].shift(-1).fillna(0)

    # Вес магазина (Target Encoding)
    train_history = data[data['Date'].dt.year < 2015]
    store_profile = train_history.groupby(['Store', 'DayOfWeek'])['Sales'].mean().reset_index(name='Store_Day_Avg')
    data = data.merge(store_profile, on=['Store', 'DayOfWeek'], how='left')

    # Заполнение пустот
    fill_val = data['Sales'].mean()
    for col in ['Sales_Lag_7', 'Rolling_7', 'Store_Day_Avg']:
        data[col] = data[col].fillna(fill_val)

    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Year'] = data['Date'].dt.year

    return data

data = load_and_boost_ultra()

# Наборы признаков
features_base = ['Promo', 'DayOfWeek']
features_full = ['Promo', 'Promo_Tomorrow', 'Temp', 'Rain', 'DayOfWeek', 'Month', 'Day', 'Store_Day_Avg', 'Sales_Lag_7', 'Rolling_7']

train_set = data[data['Year'] < 2015].copy()
test_set = data[data['Year'] == 2015].copy()

# ==========================================
# 2. ОБУЧЕНИЕ МОДЕЛЕЙ
# ==========================================
print("🧠 Обучение моделей (Градиентный бустинг)...")
y_train_log = np.log1p(train_set['Sales'])

# Базовая модель
m_base = HistGradientBoostingRegressor(max_iter=100, random_state=42)
m_base.fit(train_set[features_base], y_train_log)
p_base = np.expm1(m_base.predict(test_set[features_base]))

# Продвинутая модель
m_full = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=12, random_state=42)
m_full.fit(train_set[features_full], y_train_log)
p_full = np.expm1(m_full.predict(test_set[features_full]))

test_set['Preds'] = p_full

# ==========================================
# 3. ВЫВОД В ТЕРМИНАЛ (СРАВНЕНИЕ)
# ==========================================
r2_b, r2_f = r2_score(test_set['Sales'], p_base), r2_score(test_set['Sales'], p_full)
mae_b, mae_f = mean_absolute_error(test_set['Sales'], p_base), mean_absolute_error(test_set['Sales'], p_full)
improvement = ((mae_b - mae_f) / mae_b) * 100

print("\n" + "═" * 65)
print("📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ (ULTRA PRECISION REPORT)")
print("═" * 65)
print(f"{'Метрика':<25} | {'Базовая':<15} | {'Продвинутая':<15}")
print("-" * 65)
print(f"{'Точность (R2 Score)':<25} | {r2_b:<15.2%} | \033[92m{r2_f:<15.2%}\033[0m")
print(f"{'Средняя ошибка (MAE)':<25} | {mae_b:<15.2f} | \033[92m{mae_f:<15.2f}\033[0m")
print("-" * 65)
print(f"🚀 ИТОГ: Точность выросла на {r2_f - r2_b:+.2%}")
print(f"📉 Ошибка (MAE) снизилась на \033[92m{improvement:.2f}%\033[0m")
print("═" * 65)

# ==========================================
# 4. ДАШБОРД (3 ГРАФИКА)
# ==========================================
# Агрегируем по дням для плавности тренда
daily = test_set.groupby('Date')[['Sales', 'Preds']].mean().reset_index()

# Расчет важности факторов
print("📊 Расчет важности признаков...")
result = permutation_importance(m_full, test_set[features_full], np.log1p(test_set['Sales']), n_repeats=3, random_state=42)
importance_df = pd.DataFrame({'Feature': features_full, 'Weight': result.importances_mean}).sort_values('Weight')

fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=("Динамика спроса 2015: Факт vs Прогноз", "Важность факторов влияния", "Распределение погрешности"),
    vertical_spacing=0.1,
    row_heights=[0.5, 0.25, 0.25]
)

# График 1: Тренды
fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Sales'], name='Реальность (Среднее)', line=dict(color='#2ECC71', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Preds'], name='Прогноз модели', line=dict(color='#E67E22', width=3, dash='dash')), row=1, col=1)

# График 2: Важность
fig.add_trace(go.Bar(x=importance_df['Weight'], y=importance_df['Feature'], orientation='h', marker_color='#00CC96', name='Влияние'), row=2, col=1)

# График 3: Погрешность
fig.add_trace(go.Histogram(x=test_set['Sales'] - test_set['Preds'], name='Ошибка', marker_color='#EF553B', opacity=0.7), row=3, col=1)

fig.update_layout(height=1000, title_text=f"<b>ROSSMANN BI DASHBOARD 2015 (R2: {r2_f:.2%})</b>", template="plotly_dark", showlegend=True)
fig.show()

# ==========================================
# 5. МОДУЛЬ ПРИНЯТИЯ РЕШЕНИЙ
# ==========================================
try:
    print("\n" + "🚀 МОДУЛЬ ПРИНЯТИЯ РЕШЕНИЙ")
    u_promo = int(input("Акция активна? (1-да, 0-нет): "))
    u_temp = float(input("Температура воздуха (°C): "))
    u_rain = int(input("Ожидается дождь? (1-да, 0-нет): "))
    u_day = int(input("День недели (0-6): "))
    u_price = float(input("Средний чек: "))

    # Заглушки для лагов при ручном вводе (используем средние значения)
    u_input = pd.DataFrame([[u_promo, 0, u_temp, u_rain, u_day, 5, 15, data['Store_Day_Avg'].mean(), data['Sales_Lag_7'].mean(), data['Rolling_7'].mean()]], columns=features_full)
    u_pred = np.expm1(m_full.predict(u_input))

    print("\n" + "—" * 40)
    print(f"ПРОГНОЗ СПРОСА: {u_pred:,.0f} ед.")
    print(f"ОЖИДАЕМАЯ ВЫРУЧКА: {u_pred * u_price:,.2f}")
    print("—" * 40)
except Exception as e:
    print(f"❌ Ошибка ввода: {e}")
