from flask import Flask,render_template,request,flash,url_for
from flask import redirect
import os
from werkzeug.utils import secure_filename
# from detect import recognition_testing
import cv2
import numpy as np
from find_details import getdetails
from sendfine import send_fine
from flask_ngrok import run_with_ngrok
from yolo import recognise
from core.plate_recognition import recognize_plate

app = Flask(__name__)
run_with_ngrok(app) 

app.secret_key = "majorproject"

app.config["IMAGE_UPLOADS"] = "./static/uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024


plate_number = None
mobile_number = None


def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False



@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]


            if image.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image.filename):
                filename = secure_filename(image.filename)

                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                upload_path = "./static/uploads/"+filename

                result = recognise(upload_path)
                temp_result = result['results']
                bboxes = temp_result[0]['box']
                xmin,ymin,xmax,ymax = bboxes['xmin'],bboxes['ymin'],bboxes['xmax'],bboxes['ymax']

                img = cv2.imread(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                c1, c2 = (xmin, ymin), (xmax, ymax)


                temp_img = img
         

                cv2.rectangle(temp_img, c1, c2, (0,0,255), 5)

                tempfilename = os.path.splitext(os.path.basename(filename))[0]

                detectfilename = tempfilename+'_detected_plate.png'

                cv2.imwrite(os.path.join(app.config["IMAGE_UPLOADS"], detectfilename), temp_img)

                crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

                cropfilename = tempfilename+'_crop.png'

                cv2.imwrite(os.path.join(app.config["IMAGE_UPLOADS"], cropfilename), crop_img)

                #cv2.imshow('output', img)
                #cv2.waitKey(0)

                plate = temp_result[0]['plate']
                plate = plate.upper()

                global plate_number

                plate_number = plate
                print(plate)


                print("Image saved")

                print(filename)

               

                #return render_template('detect_result.html',orignal_img = filename,detected_img = 'detected_plate.png',crop_img = 'crop.png',plate_num = plate)

                return redirect(url_for('detected',orignalimg = filename,detectedimg = detectfilename,cropimg = cropfilename,platenum = plate))

            else:
                print("That file extension is not allowed")
                return redirect(request.url)

    return render_template("upload_image.html")


@app.route('/detected')
def detected():

	# orignal_img = request.args.get('orignalimg')
	# detected_img = request.args.get('detectedimg')
	# crop_img = request.args.get('cropimg')
	# plate_num = request.args.get('platenum')
	return render_template('detect_result.html',orignal_img = request.args.get('orignalimg'),detected_img = request.args.get('detectedimg'),crop_img = request.args.get('cropimg'),plate_num = request.args.get('platenum'))


@app.route('/details')
def details():
    try:
        details = getdetails(plate_number)
    except:
        return render_template("error_page.html",message  = "No Data Found")

    else:
        global mobile_number
        mobile_number = details[1]
        return render_template('details.html',owner_name = details[0],address = details[2],mobile_num = details[1])


# @app.route('/generate_ticket')
# def generate_ticket():

#     now = datetime.now()
#     current_time = now.strftime("%H:%M:%S")
#     print("Current Time =", current_time)
#     return render_template('ticket.html',details = details)


@app.route('/sendfine')
def sendfine():
    msg = send_fine(mobile_number)
    check = msg.find('successfully')
    print(check)
    if(check!=-1):
        return render_template("sent_success.html")
    else:
        return render_template("error_page.html",message="Error, message not sent")


if __name__=="__main__":
	app.run() 