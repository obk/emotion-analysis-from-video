fetch('http://localhost:5000/start',{
    method: 'POST',
})
    .then(response => response.json())
    .then(json => {
        let analysis = [];
        let start = null;
        let end = null;
        let prev_emotion = null;

        for (let [second, data] of Object.entries(json)) {
            let emotion = data.dominant_emotion;
            if (start === null) {  // starting point
                start = second;
                prev_emotion = emotion;
            } else if (emotion !== prev_emotion) {  // emotion has changed
                end = second - 1;
                analysis.push(`User is ${prev_emotion} between the seconds ${start} and ${end}`);
                start = second;
                prev_emotion = emotion;
            }
        }

        // handle the last interval
        end = Object.keys(json).pop();
        analysis.push(`User is ${prev_emotion} between the seconds ${start} and ${end}`);

        console.log(analysis.join('\n'));
    })
    .catch(error => console.error('Error:', error));