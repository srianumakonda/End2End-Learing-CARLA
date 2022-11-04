import glob

image = glob.glob("output/*.jpg")
vals = [line for line in open("output/data.txt")]
print(len(image), len(vals))
