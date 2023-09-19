import numpy as np 
import cv2 


data = np.random.randint(0,20,(100,2))

map_img = cv2.imread("/home/aeelab/Desktop/personDev/noPerson.jpeg")
width, height, _ = map_img.shape 

for coord in data:
    x, y = coord
    x = int(( x / 20 ) * width)
    y = int(( y / 20 ) * height)
    cv2.circle( map_img, (x,y), 20, (255,0,0), -1  )

heatmap_image = np.zeros((width,height,1), np.uint8) 

heatmap_image = cv2.distanceTransform(heatmap_image, cv2.DIST_L2, 5)
heatmap_image = heatmap_image * 1
heatmap_image = np.uint8(heatmap_image)
heatmap_image = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_JET).astype(np.uint8)

print(heatmap_image.shape, map_img.shape)

fin_img = cv2.addWeighted(heatmap_image, 0.5, map_img, 0.5, 0)


while True:
    cv2.imshow("map", fin_img )
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()