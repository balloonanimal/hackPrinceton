from converter import get_pictures
import matplotlib.pyplot as plt

image = get_pictures(['A'])[0]

print(image.shape)
plt.axis('off')
plt.imshow(image, cmap='gray')
plt.show()
