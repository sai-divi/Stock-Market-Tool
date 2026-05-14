// ==UserScript==
// @name         Overlay Predictor
// @namespace    http://localhost:8765/
// @version      1.0
// @description  Shows BUY/SELL/HOLD predictions on Yahoo Finance via local ML server
// @author       OverlayPredictor
// @match        https://finance.yahoo.com/quote/*
// @icon         data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    var API = 'http://localhost:8765';
    var ticker = '';
    var lastPred = null;
    var connected = false;

    function getTicker() {
        var m = window.location.pathname.match(/\/quote\/([A-Za-z0-9.]+)/);
        return m ? m[1].toUpperCase() : '';
    }

    function getPrice() {
        // Try multiple selectors for Yahoo Finance price element
        var sel = [
            'fin-streamer[data-field="regularMarketPrice"]',
            '[data-test="qsp-price"]',
            '.livePrice',
            'span[data-reactid*="price"]'
        ];
        for (var i = 0; i < sel.length; i++) {
            var el = document.querySelector(sel[i]);
            if (el) {
                var txt = el.getAttribute('value') || el.textContent || el.innerText || '';
                txt = txt.replace(/[^0-9.]/g, '');
                var n = parseFloat(txt);
                if (!isNaN(n) && n > 0) return n;
            }
        }
        return null;
    }

    function createUI() {
        var d = document.createElement('div');
        d.id = 'op-overlay';
        d.style.cssText = 'position:fixed;top:80px;right:20px;z-index:999999;' +
            'background:rgba(13,13,13,0.85);color:#fff;font-family:Arial,sans-serif;' +
            'font-size:13px;padding:12px 16px;border-radius:8px;min-width:180px;' +
            'border:1px solid #333;box-shadow:0 4px 20px rgba(0,0,0,0.5);' +
            'backdrop-filter:blur(4px);text-align:center;';
        d.innerHTML = '<div style="font-size:11px;color:#888;margin-bottom:4px;">Overlay Predictor</div>' +
            '<div id="op-ticker" style="font-size:18px;font-weight:bold;margin-bottom:2px;">--</div>' +
            '<div id="op-rec" style="font-size:24px;font-weight:bold;margin:4px 0;">WAIT</div>' +
            '<div id="op-conf" style="font-size:13px;color:#888;"></div>' +
            '<div id="op-price" style="font-size:12px;color:#aaa;margin-top:4px;"></div>' +
            '<div id="op-news" style="font-size:11px;color:#888;margin-top:4px;"></div>' +
            '<div id="op-reasons" style="font-size:10px;color:#666;margin-top:6px;text-align:left;"></div>' +
            '<div id="op-status" style="font-size:9px;color:#555;margin-top:6px;"></div>';
        document.body.appendChild(d);
    }

    function updateUI(pred) {
        var recEl = document.getElementById('op-rec');
        var confEl = document.getElementById('op-conf');
        var priceEl = document.getElementById('op-price');
        var tickerEl = document.getElementById('op-ticker');
        var reasonsEl = document.getElementById('op-reasons');
        var statusEl = document.getElementById('op-status');
        var newsEl = document.getElementById('op-news');

        if (!recEl) return;

        tickerEl.textContent = ticker;

        if (!pred || pred.error) {
            recEl.textContent = 'ERR';
            recEl.style.color = '#ff1744';
            confEl.textContent = pred ? pred.error : 'No connection';
            statusEl.textContent = 'Connecting to ' + API + '...';
            return;
        }

        connected = true;
        var dir = pred.prediction || 'HOLD';
        var conf = pred.confidence || 0;
        var isBuy = dir === 'BUY';
        var isSell = dir === 'SELL';
        var color = isBuy ? '#00c853' : (isSell ? '#ff1744' : '#ffd600');
        var arrow = isBuy ? '\u25B2' : (isSell ? '\u25BC' : '\u25CB');

        recEl.textContent = arrow + ' ' + dir;
        recEl.style.color = color;
        confEl.textContent = 'Confidence: ' + conf.toFixed(0) + '%';
        confEl.style.color = color;

        var lp = pred.latest_price || 0;
        priceEl.textContent = '$' + lp.toFixed(2);

        var ns = pred.news_sentiment || 0;
        newsEl.textContent = 'News: ' + (ns > 0.15 ? '[+]' : ns < -0.15 ? '[-]' : '[=]') +
            ' (' + ns.toFixed(2) + ')';

        var reasons = pred.reasons || [];
        reasonsEl.innerHTML = reasons.map(function(r) {
            return '<div>\u2022 ' + r + '</div>';
        }).join('');

        statusEl.textContent = 'Samples: ' + (pred.samples || 0) +
            (pred.trained ? ' | ML active' : ' | Rule-based');
    }

    function tick() {
        var newTicker = getTicker();
        if (newTicker && newTicker !== ticker) {
            ticker = newTicker;
            // Train on new ticker
            fetch(API + '/api/train?ticker=' + ticker, {mode:'cors'}).catch(function(){});
        }

        var price = getPrice();
        if (!ticker || !price) {
            updateUI(null);
            setTimeout(tick, 1000);
            return;
        }

        fetch(API + '/api/predict', {
            method: 'POST',
            mode: 'cors',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ticker: ticker, price: price})
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            lastPred = data;
            updateUI(data);
        })
        .catch(function(err) {
            connected = false;
            updateUI(null);
        });

        setTimeout(tick, 1000);
    }

    // Wait for page to load, then start
    if (document.readyState === 'complete') {
        createUI();
        tick();
    } else {
        window.addEventListener('load', function() {
            createUI();
            tick();
        });
    }
})();
