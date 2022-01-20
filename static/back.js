
var s = window.screen;
var width = q.width = s.width;
var height = q.height = s.height;
var letters = Array(256).join(10).split('');

var draw = function () {
    // q.getContext('2d').fillStyle = 'rgba(214, 179, 24, 0.664)'; //背景色
    q.getContext('2d').fillStyle = 'rgba(15, 35, 80, 0.2)'; //背景色
    q.getContext('2d').fillRect(0, 0, width, height);
    // q.getContext('2d').fillStyle = 'rgba(0,0,0,)';
    q.getContext('2d').fillStyle = '#5383c3'; //文字色
    // q.getContext('2d').fillStyle = '#e9dfe5'; //文字色 桜鼠さくらねず
    letters.map(function (y_pos, index) {
        text = String.fromCharCode(3e4 + Math.random() * 33);
        x_pos = index * 10;
        q.getContext('2d').fillText(text, x_pos, y_pos);
        letters[index] = (y_pos > 758 + Math.random() * 1e4) ? 0 : y_pos + 10;
    });
};
setInterval(draw, 50);