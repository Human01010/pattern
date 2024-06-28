from PIL import Image
# 打开图像文件
image = Image.open('/pattern_detection/Project/char_tmpl/9.jpg')
# 左右翻转图像
flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
# 保存翻转后的图像
flipped_image.save('C:\pyprojects\lessons\pattern_detection\Project\char_tmpl/19.jpg')
# 显示原图和翻转后的图像
image.show()
flipped_image.show()
