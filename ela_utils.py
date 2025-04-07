from PIL import Image, ImageChops, ImageEnhance

IMG_SIZE = (128, 128)

def ela_transform(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    resaved_path = "static/temp_resaved.jpg"
    original.save(resaved_path, 'JPEG', quality=quality)
    resaved = Image.open(resaved_path)
    diff = ImageChops.difference(original, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(diff).enhance(scale)
    return ela_image
