import sys
import traceback
import os
import xml.etree.ElementTree as etree
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
#sys.path.insert(0, "/Users/xxz005/propviz/")
#from propviz.drawing import draw_shapely
import matplotlib.pyplot as plt
from PIL import Image

import argparse

width=2000#1000 
height=2000#1000
channels=3
display = False

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='parsing XML folder')
  parser.add_argument('--input', dest='input_path', help='input path', default='', type=str)
  parser.add_argument('--output', dest='output_path', help='output path', default='', type=str)
  args = parser.parse_args()
  pathName = args.input_path
  output_path=args.output_path

  cnt=0

  for file in os.listdir(pathName):
    if file.endswith(".xml"):
        try:
            print('%d: %s\n' %(cnt, file))
            cnt+=1
            tree = etree.parse(pathName + file) 
            root = tree.getroot()
            fileName = (root.find('filename')).text
            polygons = []
            bboxes = []

            f = open(output_path + file,'w') 
            line = "<annotation>" + '\n'
            f.write(line)
            line = '\t\t<folder>' + "Chimney_Streetview" + '</folder>' + '\n'
            f.write(line)
            line = '\t\t<filename>' + fileName + '</filename>' + '\n'
            f.write(line)
            line = '\t\t<source>\n\t\t<database>Source</database>\n\t</source>\n'
            f.write(line)

            line = '\t<size>\n\t\t<width>'+ str(width) + '</width>\n\t\t<height>' + str(height) + '</height>\n\t'
            line += '\t<depth>' + str(channels) + '</depth>\n\t</size>'
            f.write(line)
            line = '\n\t<segmented>0</segmented>'
            f.write(line)

            for object in root.findall('object'):
                deleted = object.find('deleted').text.encode('utf-8').strip()
                if deleted=='1':
                    continue

                dians = []
                objType =(object.find('name')).text
                for plg in object.findall('polygon'):
                    for pt in plg.findall('pt'):
                            for xcoord in pt.findall('x'):
                                x1 = int(xcoord.text)
                            for ycoord in pt.findall('y'):
                                y1 = int(ycoord.text)
                            dians.append((x1, y1))
                polygon = Polygon(dians)

                xmin, ymin, xmax, ymax = polygon.bounds
                bbox = Polygon([(xmin, ymin),(xmax, ymin),(xmax, ymax),(xmin, ymax)])
                bboxes.append(bbox)

                line = '\n\t<object>'
                line += '\n\t\t<name>' + objType + '</name>\n\t\t<pose>Unspecified</pose>'
                line += '\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>'
                line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(int(xmin)) + '</xmin>'
                line += '\n\t\t\t<ymin>' + str(int(ymin)) + '</ymin>'
                line += '\n\t\t\t<xmax>' + str(int(xmax)) + '</xmax>'
                line += '\n\t\t\t<ymax>' + str(int(ymax)) + '</ymax>'
                line += '\n\t\t</bndbox>'
                line += '\n\t</object>\n'     
                f.write(line)
            f.write("</annotation>")
            f.close()               

        #    regions = MultiPolygon(bboxes)
        #    if display:
        #        imageName = imagePath + file.split(".xml")[0]+".jpg"
        #        out_name  = output_path+ file.split(".xml")[0]+"-bbox.jpg"
	#		print(out_name)
        #        img = Image.open(imageName)
        #        ax = plt.figure().add_subplot(111)
        #        plt.axis('off')
        #        ax.imshow(img)
        #        draw_shapely(ax, regions, edgecolor="red")
	#	plt.show()
        #        ax.figure.savefig(out_name, bbox_inches='tight')
        #        plt.close()
            
        except:
            print("Exception occured")
