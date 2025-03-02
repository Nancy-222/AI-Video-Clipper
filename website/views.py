from flask import Blueprint, render_template, request, redirect , jsonify
import os
from website.videoclipper import process_links, process_custom_videos,stop_processing

#Blueprint is like the map of the website, it contains the names of the urls or routes
views = Blueprint('views', __name__)


@views.route('/', methods=['GET','POST']) #name of the variable views
def home():
    if request.method == 'POST':
        if 'videourl' in request.form:
            videourl = request.form['videourl']
            prompt = request.form['prompt']
            fps =  float(request.form['fps'])
            if(not videourl or not prompt or not fps):
                return jsonify({"error":"You should fill everything"})

            videourls = [url.strip() for url in videourl.split(",")]
            print("Processing...")
            process_links(videourls,prompt,fps)
            videos = os.listdir(os.path.join('website',os.path.join('static','results')))
            response_data = {'videos': videos}
            return jsonify(response_data)
        elif 'videos' in request.files:
            videos = request.files.getlist('videos')
            prompt = request.form['prompt']
            fps =  float(request.form['fps'])
            print(request.files)
            print(videos[0])
            if(not videos[0] or not prompt or not fps):
                return jsonify({"error":"You should fill everything"})
            videos_names = []
            for video in videos:
                video.save("./website/videos/" + video.filename)
                videos_names.append(video.filename)

            print("Processing...")
            process_custom_videos(videos_names,prompt,fps)
            videos = os.listdir(os.path.join('website',os.path.join('static','results')))
            response_data = {'videos': videos}
            return jsonify(response_data)

    return render_template("index.html")

@views.route('/cancel', methods=['POST'])
def cancel():
    response_data = {}
    if request.method == 'POST':
        stop_processing()
        response_data = {'message':'Processing stopped successfully'}
    return jsonify(response_data)

@views.route('/HowToUse')
def how_to_use():
    return render_template("how_to_use.html")


@views.route('/About')
def about():
    return render_template("about.html")
