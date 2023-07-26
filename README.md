# Final-Project-Analytics-middle
# Исследование для компании «Мегафон».

<p align="center"><img src="/imgs/Algoritm.png" width="500" alt="Алгоритм опроса"></p>

В ходе опроса компания «Мегафон» предложила своим клиентам оценить уровень удовлетворённости качеством связи по десятибалльной шкале (где 10 — это «отлично», а 1 — «ужасно»). Если клиент оценивал качество связи на 9 или 10 баллов, опрос заканчивался. Если клиент ставил оценку ниже 9, задавался второй вопрос — о причинах неудовлетворённости качеством связи с предоставленными пронумерованными вариантами ответа. Ответ можно было дать в свободном формате или перечислить номера ответов через запятую.
megafon.csv содержит следующие значения:

     user_id — идентификатор абонента;
     Q1 — ответ на первый вопрос;
     Q2 — ответ на второй вопрос;
     Total Traffic(MB) — объем трафика передачи данных 1 ;
     Downlink Throughput(Kbps) — средняя скорость «к абоненту» 2 ;
     Uplink Throughput(Kbps)— средняя скорость «от абонента» 3 ;
     Downlink TCP Retransmission Rate(%) — частота переотправок пакетов «к абоненту» 4 ;
     Video Streaming Download Throughput(Kbps) — скорость загрузки потокового видео 5 ;
     Video Streaming xKB Start Delay(ms) — задержка старта воспроизведения видео 6 ;
     Web Page Download Throughput(Kbps) — скорость загрузки web-страниц через браузер 7 ;
     Web Average TCP RTT(ms) — пинг при просмотре web-страниц8 .

1 — Насколько активно абонент использует мобильный интернет.
2 — Считается по всему трафику передачи данных.
3 — Считается по всему трафику передачи данных.
4 — Чем выше, тем хуже. Если в канале возникает ошибка, пакет переотправляется. Снижается полезная скорость.
5 — Чем выше, тем лучше — меньше прерываний и лучше качество картинки.
6 — Сколько времени пройдёт между нажатием на кнопку Play и началом воспроизведения видео. Чем меньше это время, тем быстрее начинается воспроизведение.
7 — Чем выше, тем лучше.
8 — Чем меньше, тем лучше — быстрее загружаются web-страницы.

Первый технический показатель представлен как сумма за период в одну неделю перед участием в опросе. Остальные технические показатели отображают среднее значение по данному признаку за период в одну неделю перед участием в опросе.
## Цель исследования
Определить влияние технических параметров связи у абонентов на поставленную ими оценку по первому вопросу и на ответ по второму вопросу.
## Создание виртуальной среды
Создадим папку под проект. Назовите ее как вам будет удобно. Клонируйте этот репозиторий в созданную папку: https://github.com/mic21053/Final-Project-Analytics-middle  
Далее создаем виртуальную среду:
<pre>
python -m venv <название вашей виртуальной среды>
</pre>
Активируем её:
<pre>
source <название вашей виртуальной среды>/bin/activate
</pre>
Создаем зависимости и добавляем виртуальную среду в ядро jupyter notebook:
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=<название вашей виртуальной среды>
</pre>
Запускаем jupyter notebook и выбираем в качестве ядра наше созданное виртуальное окружение. Например на скриншоте ниже - это ядро tfod.

<p align="center"><img src="/imgs/Jupyter_view.png" width="500" alt="Ядро"></p>

В дальнейшем не забываем выходить из виртуальной среды командой:
<pre>
deactivate
</pre>
