// script.js

const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');

startBtn.addEventListener('click', function () {
    window.location.href = "/video";
});

stopBtn.addEventListener('click', function () {
    window.location.href = "/";
});
