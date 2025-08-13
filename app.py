import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import os
import json
import sqlite3
from transformers import (
    AutoImageProcessor, AutoModel
)
from tqdm import tqdm
import faiss
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash, g, jsonify
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import uuid
import base64

# --- Добавляем эту строку для решения ошибки OpenMP ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Необходим для сессий и flash-сообщений

# --- Настройка моделей и устройства ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

try:
    print("Загрузка Fashion-модели...")
    fashion_processor = AutoImageProcessor.from_pretrained("yainage90/fashion-image-feature-extractor")
    fashion_model = AutoModel.from_pretrained("yainage90/fashion-image-feature-extractor").to(device)
except Exception as e:
    print(f"Ошибка загрузки моделей: {e}")
    exit()


def get_db():
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE,
                      password TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS products
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      name TEXT,
                      image_path TEXT,
                      quantity INTEGER,
                      wholesale_price REAL,
                      retail_price REAL,
                      fashion_embedding TEXT,
                      filename TEXT,
                      FOREIGN KEY(user_id) REFERENCES users(id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS sales
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      item_id INTEGER,
                      quantity_sold INTEGER,
                      retail_price REAL,
                      profit REAL,
                      timestamp TEXT,
                      FOREIGN KEY(user_id) REFERENCES users(id),
                      FOREIGN KEY(item_id) REFERENCES products(id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS candidates
                     (user_id INTEGER,
                      item_id INTEGER,
                      score REAL,
                      PRIMARY KEY(user_id, item_id),
                      FOREIGN KEY(user_id) REFERENCES users(id),
                      FOREIGN KEY(item_id) REFERENCES products(id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS session_data
                     (user_id INTEGER PRIMARY KEY,
                      sale_image_base64 TEXT,
                      candidate_page INTEGER DEFAULT 0,
                      selected_item_id INTEGER,
                      FOREIGN KEY(user_id) REFERENCES users(id))''')
        conn.commit()


def get_features(model, processor, image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(
                dim=1)
        features /= features.norm(p=2, dim=-1, keepdim=True)
        return features.flatten().cpu().numpy()
    except Exception as e:
        print(f"Ошибка обработки изображения {image_path}: {e}")
        return None


def get_user_id(username):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        return row['id'] if row else None


def get_name(user_id, item_id):
    with get_db() as conn:
        c = conn.cursor()
        row = c.execute("SELECT name FROM products WHERE id = ? AND user_id = ?", (item_id,user_id)).fetchone()
        print(row)
        row= dict(row)
        return row.get('name')
def add_user(username, password):
    with get_db() as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return c.lastrowid
        except sqlite3.IntegrityError:
            return None


def get_user_by_username(username):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        return dict(row) if row else None


def add_product(user_id, name, image_path, quantity, wholesale_price, retail_price, fashion_embedding, filename):
    fashion_json = json.dumps(fashion_embedding.tolist())
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""INSERT INTO products (user_id, name, image_path, quantity, wholesale_price, retail_price, fashion_embedding, filename)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                  (user_id, name, image_path, quantity, wholesale_price, retail_price, fashion_json, filename))
        conn.commit()
        return c.lastrowid


def get_products(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM products WHERE user_id = ?", (user_id,))
        rows = c.fetchall()
        products = []
        for row in rows:
            item = dict(row)
            item['fashion_embedding'] = np.array(json.loads(item['fashion_embedding']))
            products.append(item)
        return products


def get_product(user_id, product_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM products WHERE id = ? AND user_id = ?", (product_id, user_id))
        row = c.fetchone()
        if row:
            item = dict(row)
            item['fashion_embedding'] = np.array(json.loads(item['fashion_embedding']))
            return item
        return None


def update_product(user_id, product_id, name, quantity, wholesale_price, retail_price, image_path=None, fashion_embedding=None, filename=None):
    with get_db() as conn:
        c = conn.cursor()
        if image_path:
            fashion_json = json.dumps(fashion_embedding.tolist())
            c.execute("""UPDATE products SET name = ?, quantity = ?, wholesale_price = ?, retail_price = ?, image_path = ?, fashion_embedding = ?, filename = ?
                         WHERE id = ? AND user_id = ?""",
                      (name, quantity, wholesale_price, retail_price, image_path, fashion_json, filename, product_id, user_id))
        else:
            c.execute("""UPDATE products SET name = ?, quantity = ?, wholesale_price = ?, retail_price = ?
                         WHERE id = ? AND user_id = ?""",
                      (name, quantity, wholesale_price, retail_price, product_id, user_id))
        # Update profits if wholesale changed
        c.execute("UPDATE sales SET profit = (retail_price - ?) * quantity_sold WHERE item_id = ? AND user_id = ?", (wholesale_price, product_id, user_id))
        conn.commit()


def delete_product(user_id, product_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT image_path FROM products WHERE id = ? AND user_id = ?", (product_id, user_id))
        row = c.fetchone()
        if row:
            image_path = row['image_path']
            if os.path.exists(image_path):
                os.remove(image_path)
        c.execute("DELETE FROM products WHERE id = ? AND user_id = ?", (product_id, user_id))
        c.execute("DELETE FROM sales WHERE item_id = ? AND user_id = ?", (product_id, user_id))
        c.execute("DELETE FROM candidates WHERE item_id = ? AND user_id = ?", (product_id, user_id))
        conn.commit()


def add_sale(user_id, item_id, quantity_sold, retail_price, profit, timestamp):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""INSERT INTO sales (user_id, item_id, quantity_sold, retail_price, profit, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (user_id, item_id, quantity_sold, retail_price, profit, timestamp))
        conn.commit()
        return c.lastrowid


def get_sales(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM sales WHERE user_id = ?", (user_id,))
        return [dict(row) for row in c.fetchall()]


def get_sale(user_id, sale_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM sales WHERE id = ? AND user_id = ?", (sale_id, user_id))
        row = c.fetchone()
        return dict(row) if row else None


def update_sale(user_id, sale_id, quantity_sold, retail_price, profit, timestamp):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""UPDATE sales SET quantity_sold = ?, retail_price = ?, profit = ?, timestamp = ?
                     WHERE id = ? AND user_id = ?""",
                  (quantity_sold, retail_price, profit, timestamp, sale_id, user_id))
        conn.commit()


def delete_sale_db(user_id, sale_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT item_id, quantity_sold FROM sales WHERE id = ? AND user_id = ?", (sale_id, user_id))
        row = c.fetchone()
        if row:
            item_id = row['item_id']
            quantity_sold = row['quantity_sold']
            c.execute("UPDATE products SET quantity = quantity + ? WHERE id = ? AND user_id = ?", (quantity_sold, item_id, user_id))
        c.execute("DELETE FROM sales WHERE id = ? AND user_id = ?", (sale_id, user_id))
        conn.commit()


def set_candidates(user_id, candidates):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM candidates WHERE user_id = ?", (user_id,))
        for item_id, score in candidates:
            c.execute("INSERT INTO candidates (user_id, item_id, score) VALUES (?, ?, ?)", (user_id, item_id, score))
        conn.commit()


def get_candidates(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT item_id, score FROM candidates WHERE user_id = ?", (user_id,))
        return [(row['item_id'], row['score']) for row in c.fetchall()]


def set_session_data(user_id, sale_image_base64=None, candidate_page=None, selected_item_id=None):
    with get_db() as conn:
        c = conn.cursor()
        if sale_image_base64 is not None:
            c.execute("UPDATE session_data SET sale_image_base64 = ? WHERE user_id = ?", (sale_image_base64, user_id))
        if candidate_page is not None:
            c.execute("UPDATE session_data SET candidate_page = ? WHERE user_id = ?", (candidate_page, user_id))
        if selected_item_id is not None:
            c.execute("UPDATE session_data SET selected_item_id = ? WHERE user_id = ?", (selected_item_id, user_id))
        if c.rowcount == 0:
            c.execute("""INSERT INTO session_data (user_id, sale_image_base64, candidate_page, selected_item_id)
                         VALUES (?, ?, ?, ?)""", (user_id, sale_image_base64, candidate_page or 0, selected_item_id))
        conn.commit()


def get_session_data(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM session_data WHERE user_id = ?", (user_id,))
        row = c.fetchone()
        if row:
            return dict(row)
        return {'sale_image_base64': None, 'candidate_page': 0, 'selected_item_id': None}


def clear_session_data(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM session_data WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM candidates WHERE user_id = ?", (user_id,))
        conn.commit()


def find_candidates(image_path):
    user_id = get_user_id(g.user)
    products = get_products(user_id)
    if not products:
        flash("База данных пуста. Пожалуйста, добавьте товары.")
        return None
    try:
        new_fashion_embedding = get_features(fashion_model, fashion_processor, image_path)
        if new_fashion_embedding is None:
            flash("Не удалось обработать изображение для поиска.")
            return None
    except Exception as e:
        print(f"Ошибка при поиске кандидатов: {e}")
        flash(f"Ошибка при обработке изображения: {e}")
        return None

    k = min(20, len(products))
    fashion_dim = len(new_fashion_embedding)
    fashion_index = faiss.IndexFlatIP(fashion_dim)

    fashion_embeddings = np.stack([item['fashion_embedding'] for item in products])

    fashion_index.add(fashion_embeddings)

    fashion_similarities, fashion_indices = fashion_index.search(np.array([new_fashion_embedding]), k)

    candidate_scores = {}
    for i in range(k):
        fashion_idx = fashion_indices[0][i]
        fashion_id = products[fashion_idx]['id']
        candidate_scores[fashion_id] = float(fashion_similarities[0][i])

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Вы должны войти в систему.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.before_request
def before_request():
    g.user = session.get('user')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        if add_user(username, password) is None:
            flash('Имя пользователя уже существует.')
        else:
            session['user'] = username
            flash('Регистрация успешна.')
            return redirect(url_for('home'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        user = get_user_by_username(username)
        if user and check_password_hash(user['password'], request.form['password']):
            session['user'] = username
            flash('Вход успешен.')
            return redirect(url_for('home'))
        flash('Неверные учетные данные.')
    return render_template('login.html')


@app.route('/logout')
def logout():
    user_id = get_user_id(session.pop('user', None))
    if user_id:
        clear_session_data(user_id)
    flash('Вы вышли из системы.')
    return redirect(url_for('login'))


@app.route('/')
@login_required
def home():
    return render_template('add_item.html')


@app.route('/add', methods=['POST'])
@login_required
def add_item():
    user_id = get_user_id(g.user)
    if not user_id:
        return redirect(url_for('login'))

    name = request.form.get('name')
    wholesale_price = request.form.get('wholesale_price')
    retail_price = request.form.get('retail_price')
    quantity = request.form.get('quantity')
    image_file = request.files.get('image')

    if not all([name, wholesale_price, retail_price, quantity, image_file and image_file.filename]):
        flash('Все поля обязательны, включая изображение.')
        return redirect(url_for('home'))

    try:
        wholesale_price = float(wholesale_price)
        retail_price = float(retail_price)
        quantity = int(quantity)
    except ValueError:
        flash('Цены и количество должны быть числовыми.')
        return redirect(url_for('home'))

    ext = os.path.splitext(image_file.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        flash('Поддерживаются только изображения JPG и PNG.')
        return redirect(url_for('home'))

    filename = f"{uuid.uuid4().hex}{ext}"
    image_path = os.path.join('demo_images', filename)
    image_file.save(image_path)

    fashion_embedding = get_features(fashion_model, fashion_processor, image_path)
    if fashion_embedding is None:
        os.remove(image_path)
        flash('Ошибка: Не удалось обработать изображение.')
        return redirect(url_for('home'))

    product_id = add_product(user_id, name, image_path, quantity, wholesale_price, retail_price, fashion_embedding, filename)
    flash('Товар успешно добавлен.')
    return redirect(url_for('home'))


@app.route('/sell', methods=['GET', 'POST'])
@login_required
def sell_item_route():
    user_id = get_user_id(g.user)
    if not user_id:
        return redirect(url_for('login'))

    session_data = get_session_data(user_id)
    if request.method == 'POST':
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename:
                ext = os.path.splitext(image_file.filename)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    flash('Поддерживаются только изображения JPG и PNG.')
                    return redirect(url_for('sell_item_route'))

                sale_image_filename = f"temp_sale_{uuid.uuid4().hex}{ext}"
                sale_image_path = os.path.join('demo_images', sale_image_filename)
                image_file.save(sale_image_path)

                candidates = find_candidates(sale_image_path)
                if candidates:
                    with open(sale_image_path, 'rb') as f:
                        sale_image_base64 = base64.b64encode(f.read()).decode('utf-8')
                    os.remove(sale_image_path)
                    set_session_data(user_id, sale_image_base64, 0)
                    set_candidates(user_id, candidates)
                    best_id, best_similarity = candidates[0]
                    best_match = get_product(user_id, best_id)
                    return render_template('sell_item.html', best_match=best_match, similarity=best_similarity,
                                           sale_image_base64=sale_image_base64, show_confirm=True)
                else:
                    os.remove(sale_image_path)
                    return redirect(url_for('sell_item_route'))

        elif 'action' in request.form:
            action = request.form['action']
            if action == 'confirm_match':
                item_id = int(request.form['item_id'])
                set_session_data(user_id, selected_item_id=item_id)
                selected_match = get_product(user_id, item_id)
                return render_template('sell_item.html', best_match=selected_match, similarity=0,
                                       sale_image_base64=session_data['sale_image_base64'], show_sale_form=True)

            elif action == 'reject_match':
                candidates = get_candidates(user_id)
                if not candidates:
                    flash("Нет доступных кандидатов для отображения.")
                    return redirect(url_for('sell_item_route'))
                page = session_data['candidate_page']
                page_size = 5
                start = page * page_size
                end = start + page_size
                alt_candidates = candidates[start:end]
                alt_matches = [(get_product(user_id, id_), score) for id_, score in alt_candidates]

                has_more = end < len(candidates)
                return render_template('sell_item.html', alt_matches=alt_matches, has_more=has_more,
                                       sale_image_base64=session_data['sale_image_base64'])

            elif action == 'next_page':
                candidates = get_candidates(user_id)
                if not candidates:
                    flash("Нет доступных кандидатов для отображения.")
                    return redirect(url_for('sell_item_route'))
                page = session_data['candidate_page'] + 1
                set_session_data(user_id, candidate_page=page)
                page_size = 5
                start = page * page_size
                end = start + page_size
                alt_candidates = candidates[start:end]
                alt_matches = [(get_product(user_id, id_), score) for id_, score in alt_candidates]
                page_2 = page > 0
                has_more = end < len(candidates)
                return render_template('sell_item.html', alt_matches=alt_matches, has_more=has_more,
                                       sale_image_base64=session_data['sale_image_base64'], page_2=page_2)

            elif action == 'prev_page':
                candidates = get_candidates(user_id)
                if not candidates:
                    flash("Нет доступных кандидатов для отображения.")
                    return redirect(url_for('sell_item_route'))
                page = max(0, session_data['candidate_page'] - 1)
                set_session_data(user_id, candidate_page=page)
                page_size = 5
                start = page * page_size
                end = start + page_size
                alt_candidates = candidates[start:end]
                alt_matches = [(get_product(user_id, id_), score) for id_, score in alt_candidates]
                page_2 = page > 0
                has_more = end < len(candidates)
                return render_template('sell_item.html', alt_matches=alt_matches, has_more=has_more,
                                       sale_image_base64=session_data['sale_image_base64'], page_2=page_2)

    return render_template('sell_item.html')


@app.route('/confirm_sale/<int:item_id>', methods=['POST'])
@login_required
def confirm_sale(item_id):
    user_id = get_user_id(g.user)
    if not user_id:
        return redirect(url_for('login'))

    try:
        quantity_sold = int(request.form['quantity_sold'])
    except ValueError:
        flash("Количество должно быть числом.")
        return redirect(url_for('sell_item_route'))

    item = get_product(user_id, item_id)
    if not item or item['quantity'] < quantity_sold:
        flash("Ошибка: Недостаточно товара или неверный ID.")
        return redirect(url_for('sell_item_route'))

    profit = (item['retail_price'] - item['wholesale_price']) * quantity_sold
    timestamp = datetime.now().isoformat()
    add_sale(user_id, item_id, quantity_sold, item['retail_price'], profit, timestamp)
    update_product(user_id, item_id, item['name'], item['quantity'] - quantity_sold, item['wholesale_price'], item['retail_price'])
    clear_session_data(user_id)
    flash('Продажа успешно подтверждена.')
    return redirect(url_for('sell_item_route'))


@app.route('/history')
@login_required
def history():
    user_id = get_user_id(g.user)
    if not user_id:
        return redirect(url_for('login'))

    sales = get_sales(user_id)
    for sale in sales:
        sale['name'] = get_name(sale.get("user_id"),sale.get("item_id"))
    print(sales)
    database = get_products(user_id)
    total_profit = sum(s['profit'] for s in sales)
    return render_template('history.html', sales_history=sales, database=database, total_profit=total_profit)


@app.route('/edit_item/<int:item_id>', methods=['GET', 'POST'])
@login_required
def edit_item(item_id):
    user_id = get_user_id(g.user)
    if not user_id:
        return redirect(url_for('login'))

    item = get_product(user_id, item_id)
    if not item:
        flash('Неверный ID товара.')
        return redirect(url_for('history'))

    if request.method == 'POST':
        name = request.form['name']
        try:
            quantity = int(request.form['quantity'])
            wholesale_price = float(request.form['wholesale_price'])
            retail_price = float(request.form['retail_price'])
        except ValueError:
            flash('Цены и количество должны быть числовыми.')
            return redirect(url_for('edit_item', item_id=item_id))
        image_path = None
        filename = None
        fashion_embedding = None
        if 'image' in request.files and request.files['image'].filename:
            image_file = request.files['image']
            ext = os.path.splitext(image_file.filename)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                flash('Поддерживаются только изображения JPG и PNG.')
                return redirect(url_for('edit_item', item_id=item_id))
            if os.path.exists(item['image_path']):
                os.remove(item['image_path'])
            filename = f"{uuid.uuid4().hex}{ext}"
            image_path = os.path.join('demo_images', filename)
            image_file.save(image_path)
            fashion_embedding = get_features(fashion_model, fashion_processor, image_path)
            if fashion_embedding is None:
                os.remove(image_path)
                flash('Ошибка обработки изображения.')
                return redirect(url_for('edit_item', item_id=item_id))
            # Update candidates if active search
            session_data = get_session_data(user_id)
            if get_candidates(user_id) and session_data['sale_image_base64']:
                candidates = find_candidates(image_path)
                set_candidates(user_id, candidates)
                set_session_data(user_id, candidate_page=0)
        update_product(user_id, item_id, name, quantity, wholesale_price, retail_price, image_path, fashion_embedding, filename)
        flash('Товар обновлен.')
        return redirect(url_for('history'))
    return render_template('edit_item.html', item=item)


@app.route('/delete_item/<int:item_id>')
@login_required
def delete_item(item_id):
    user_id = get_user_id(g.user)
    if not user_id:
        return redirect(url_for('login'))

    delete_product(user_id, item_id)
    flash('Товар удален.')
    return redirect(url_for('history'))


@app.route('/edit_sale/<int:sale_id>', methods=['GET', 'POST'])
@login_required
def edit_sale(sale_id):
    user_id = get_user_id(g.user)
    if not user_id:
        return redirect(url_for('login'))

    sale = get_sale(user_id, sale_id)
    if not sale:
        flash('Неверный ID продажи.')
        return redirect(url_for('history'))
    name = get_name(sale['user_id'], sale['item_id'])
    item = get_product(user_id, sale['item_id'])
    if not item:
        flash('Товар для этой продажи не найден.')
        return redirect(url_for('history'))

    max_quantity = item['quantity'] + sale['quantity_sold']

    if request.method == 'POST':
        try:
            new_quantity_sold = int(request.form['quantity_sold'])
            new_retail_price = float(request.form['retail_price'])
        except ValueError:
            flash('Количество и цена должны быть числовыми.')
            return redirect(url_for('edit_sale', sale_id=sale_id))

        if new_quantity_sold < 1 or new_retail_price < 0:
            flash('Количество должно быть больше 0, а цена не отрицательной.')
            return redirect(url_for('edit_sale', sale_id=sale_id))

        if new_quantity_sold > max_quantity:
            flash('Недостаточно товара на скlade для указанного количества.')
            return redirect(url_for('edit_sale', sale_id=sale_id))

        # Revert old
        update_product(user_id, sale['item_id'], item['name'], item['quantity'] + sale['quantity_sold'], item['wholesale_price'], item['retail_price'])

        # New
        new_profit = (new_retail_price - item['wholesale_price']) * new_quantity_sold
        new_timestamp = datetime.now().isoformat()
        update_sale(user_id, sale_id, new_quantity_sold, new_retail_price, new_profit, new_timestamp)
        update_product(user_id, sale['item_id'], item['name'], item['quantity'] + sale['quantity_sold'] - new_quantity_sold, item['wholesale_price'], item['retail_price'])
        flash('Продажа успешно обновлена.')
        return redirect(url_for('history'))

    return render_template('edit_sale.html', sale=sale, max_quantity=max_quantity, sale_id=sale_id, name=name)


@app.route('/delete_sale/<int:sale_id>')
@login_required
def delete_sale(sale_id):
    user_id = get_user_id(g.user)
    if not user_id:
        return redirect(url_for('login'))

    sale = get_sale(user_id, sale_id)
    if not sale:
        flash('Неверный ID продажи.')
        return redirect(url_for('history'))

    item = get_product(user_id, sale['item_id'])
    if item:
        update_product(user_id, sale['item_id'], item['name'], item['quantity'] + sale['quantity_sold'], item['wholesale_price'], item['retail_price'])

    delete_sale_db(user_id, sale_id)
    flash('Продажа успешно удалена.')
    return redirect(url_for('history'))


@app.route('/sort_by_image', methods=['POST'])
@login_required
def sort_by_image():
    user_id = get_user_id(g.user)
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'Изображение не предоставлено.'}), 400
    image_file = request.files['image']
    ext = os.path.splitext(image_file.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        return jsonify({'error': 'Поддерживаются только изображения JPG и PNG.'}), 400

    filename = f"temp_sort_{uuid.uuid4().hex}{ext}"
    image_path = os.path.join('demo_images', filename)
    image_file.save(image_path)

    candidates = find_candidates(image_path)
    os.remove(image_path)
    if not candidates:
        return jsonify({'error': 'Не удалось найти подходящие товары.'}), 400

    sorted_items = []
    for item_id, score in candidates:
        item = get_product(user_id, item_id)
        score = round(score, 2)
        if item:
            sorted_items.append({
                'score': score,
                'id': item['id'],
                'name': item['name'],
                'quantity': item['quantity'],
                'wholesale_price': item['wholesale_price'],
                'retail_price': item['retail_price'],
                'filename': item['filename']
            })

    return jsonify({'results': sorted_items})


@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('demo_images', filename)

PORT = os.getenv("PORT")
# The following code allows you to run the application directly from the command line
# To run this, save the file as app.py and execute: python app.py
print(PORT)
if __name__ == '__main__':
  app.run(debug=True, host="0.0.0.0", port=PORT)