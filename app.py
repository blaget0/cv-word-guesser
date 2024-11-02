from flask import Flask, request, render_template, session, jsonify
import os
import cv2
from model import predict, classes
from random import choice
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    reset_game()
    return render_template('index.html', uploaded_images=[])

def reset_game():
    """Сбрасывает состояние игры."""
    session['attempts'] = 10
    session['correct_word'] = choice(classes)
    session['uploaded_images'] = []  # Список загруженных изображений
    print(session['correct_word'])

@app.route('/upload', methods=['POST'])
def upload_file():
    """Обрабатывает загрузку файла изображения."""
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла для загрузки!'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Нет выбранного файла!'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    if not os.path.exists(file_path):
        return jsonify({'error': 'Ошибка: файл не был сохранён!'}), 400

    # Уменьшаем количество попыток сразу после загрузки файла
    session['attempts'] -= 1

    try:
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Ошибка: не удалось загрузить изображение!'}), 400

        model_path = 'model.pt'
        probability = predict(file_path, model_path, session['correct_word'])
        probability = str(round(probability, 4) * 100) + "%"

        # Сохраняем информацию о загруженном изображении и его вероятности
        session['uploaded_images'].append((file.filename, probability, image_to_base64(image)))

        # Проверка состояния игры
        game_over = session['attempts'] < 1
        correct_word = session['correct_word'] if game_over else None

        return jsonify({
            'probability': probability,
            'attempts': session['attempts'],
            'game_over': game_over,
            'correct_word': correct_word,
            'uploaded_images': session['uploaded_images']
        })

    except Exception as e:
        return jsonify({'error': f'Ошибка при обработке изображения: {str(e)}'}), 500

def image_to_base64(image):
    """Преобразует изображение в base64 строку."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/guess', methods=['POST'])
def guess():
    """Обрабатывает попытку угадать слово."""
    user_guess = request.json.get('guess')
    if user_guess == session['correct_word']:
        return jsonify({'message': 'Вы выиграли!', 'game_over': True, 'correct_word': session['correct_word']})

    session['attempts'] -= 2  # Уменьшаем попытки только при неверном угадывании
    if session['attempts'] < 1:
        return jsonify({'message': 'Вы не угадали!', 'game_over': True, 'correct_word': session['correct_word']})

    return jsonify({'message': 'Попробуйте снова!', 'attempts': session['attempts'], 'game_over': False})

@app.route('/restart', methods=['POST'])
def restart():
    """Перезапускает игру."""
    reset_game()
    return jsonify({'message': 'Игра перезапущена!', 'attempts': session['attempts']})

if __name__ == '__main__':
    app.run(debug=True)
