import pytesseract

def text_recognize(img):
	custom_config = r'--oem 3 --psm 4'
	text = pytesseract.image_to_string(img, config=custom_config, lang='eng')
	return text
