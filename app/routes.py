from flask import jsonify, request, current_app as app

@app.route('/api/saludo', methods=['GET'])
def saludo():
    nombre = request.args.get('nombre', 'Mundo')
    return jsonify({'mensaje': f'Hola, {nombre}!'})