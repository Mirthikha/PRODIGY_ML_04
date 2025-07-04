from predict import predict_image

label_map = {
    0: 'Palm',
    1: 'Fist',
    2: 'Thumbs Up',
    3: 'Peace',
    4: 'Point'
}

predict_image('./test_image.png', 'my_model.h5', label_map)
