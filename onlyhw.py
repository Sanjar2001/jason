# import pandas as pd
# import json

# # Создаем данные
# data = [
#     {'name': 'Valera', 'hobbies': ['play guitar', 'programming']},
#     {'name': 'Anna', 'hobbies': ['reading', 'painting', 'yoga']},
#     {'name': 'Ivan', 'hobbies': ['football', 'cooking']}
# ]

# # Преобразуем списки хобби в JSON-строки
# for item in data:
#     item['hobbies'] = json.dumps(item['hobbies'])

# # Создаем датафрейм
# df = pd.DataFrame(data)

# # Выводим результат
# print(df)

# # Пример, как можно получить список хобби обратно
# print("\nХобби Валеры:")
# valera_hobbies = json.loads(df.loc[df['name'] == 'Valera', 'hobbies'].iloc[0])
# print(valera_hobbies)

# import json
# import sqlite3
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Инициализация модели и токенизатора
# model_name = "microsoft/DialoGPT-medium"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Подключение к базе данных
# conn = sqlite3.connect('chatbot.db')
# cursor = conn.cursor()

# # Создание таблицы, если она не существует
# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS conversations
#     (id INTEGER PRIMARY KEY, context TEXT)
# ''')

# def save_context(context):
#     # Преобразование контекста в JSON-строку
#     context_json = json.dumps(context, ensure_ascii=False)
    
#     # Сохранение контекста в базе данных
#     cursor.execute('INSERT OR REPLACE INTO conversations (id, context) VALUES (1, ?)', (context_json,))
#     conn.commit()

# def load_context():
#     # Загрузка контекста из базы данных
#     cursor.execute('SELECT context FROM conversations WHERE id = 1')
#     result = cursor.fetchone()
    
#     if result:
#         # Преобразование JSON-строки обратно в список словарей
#         return json.loads(result[0])
#     return []

# def generate_response(input_text, context):
#     # Добавление нового сообщения пользователя в контекст
#     context.append({"role": "user", "content": input_text})
    
#     # Преобразование контекста в строку для модели
#     context_text = " ".join([msg["content"] for msg in context])
    
#     # Токенизация входных данных
#     inputs = tokenizer.encode(context_text + tokenizer.eos_token, return_tensors="pt")
    
#     # Генерация ответа
#     attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=inputs.device)
#     outputs = model.generate(inputs, max_length=150, num_return_sequences=1, attention_mask=attention_mask)
    
#     # Декодирование ответа
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # Извлечение только нового ответа
#     new_response = response[len(context_text):].strip()
    
#     # Добавление ответа ассистента в контекст
#     context.append({"role": "assistant", "content": new_response})
    
#     return new_response, context

# def chat():
#     context = load_context()
    
#     print("Бот: Привет! Я готов общаться. Введите 'выход' для завершения.")
    
#     while True:
#         user_input = input("Вы: ")
        
#         if user_input.lower() == 'выход':
#             save_context(context)
#             break
        
#         response, context = generate_response(user_input, context)
#         print("Бот:", response)
        
#         # Сохранение обновленного контекста после каждого обмена сообщениями
#         save_context(context)

# if __name__ == "__main__":
#     chat()

import json
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Инициализация модели и токенизатора
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Подключение к базе данных
conn = sqlite3.connect('chatbot.db')
cursor = conn.cursor()

# Создание таблицы, если она не существует
cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations
    (id INTEGER PRIMARY KEY, context TEXT, context_len INTEGER)
''')

def save_context(context, context_len):
    # Преобразование контекста в JSON-строку
    context_json = json.dumps(context, ensure_ascii=False)
    
    # Сохранение контекста и его длины в базе данных
    cursor.execute('INSERT OR REPLACE INTO conversations (id, context, context_len) VALUES (1, ?, ?)', (context_json, context_len))
    conn.commit()

def load_context():
    # Загрузка контекста из базы данных
    cursor.execute('SELECT context, context_len FROM conversations WHERE id = 1')
    result = cursor.fetchone()
    
    if result:
        # Преобразование JSON-строки обратно в список словарей
        return json.loads(result[0]), result[1]
    return [], 0

def generate_response(input_text, context, context_len):
    # Добавление нового сообщения пользователя в контекст
    context.append({"role": "user", "content": input_text})
    context_len += len(input_text)
    
    # Преобразование контекста в строку для модели
    context_text = " ".join([msg["content"] for msg in context])
    
    # Токенизация входных данных
    inputs = tokenizer.encode(context_text + tokenizer.eos_token, return_tensors="pt")
    
    # Генерация ответа
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=inputs.device)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, attention_mask=attention_mask)
    
    # Декодирование ответа
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлечение только нового ответа
    new_response = response[len(context_text):].strip()
    
    # Добавление ответа ассистента в контекст
    context.append({"role": "assistant", "content": new_response})
    context_len += len(new_response)
    
    return new_response, context, context_len

def chat():
    context, context_len = load_context()
    
    print("Бот: Привет! Я готов общаться. Введите 'выход' для завершения.")
    
    while True:
        user_input = input("Вы: ")
        
        if user_input.lower() == 'выход':
            save_context(context, context_len)
            break
        
        response, context, context_len = generate_response(user_input, context, context_len)
        print("Бот:", response)
        print(f"Длина контекста: {context_len}")
        
        # Сохранение обновленного контекста после каждого обмена сообщениями
        save_context(context, context_len)

if __name__ == "__main__":
    chat()