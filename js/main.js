/*
    NameAI-Gender classification AI
    Copyright(C) 2018 Rupesh Sreeraman
*/
"use strict";
const SEQUENCE_LENGTH = 10;
var wordsCount = 0;
var dict = {};
var model;

function TextToIndexVector(text, seqlen) {
	var wordVec = new Array();
	//text=text.substr(text.length - 3);
	//alert(dict);
for (var i = 0; i < text.length; i++) {
  
  if (text.charAt(i) in dict) {
				wordVec.push(dict[text.charAt(i)])
				//console.log(dict[text.charAt(i)]);
			} else {
				wordVec.push(0)
			}
	
	
}
var wordVecPadded = wordVec;
var fzero=0;
if (wordVec.length<seqlen)
{
	fzero=seqlen-wordVec.length;
}
else{
	wordVecPadded=wordVec.slice(0, seqlen);
}
//padding
for (var i = 0; i < fzero; i++) 
{
	wordVec.unshift(0);
}
	//console.log(wordVecPadded);
	return wordVecPadded;
}
function LoadModel() {
$("#detectBtn").prop('disabled', true);	
NProgress.start();
$("#wordCnt").html("Loading model,please wait...");
	$.getJSON("wordindex.json", function (json) {
		
		dict = json;
		});
	model = new KerasJS.Model({
		filepath: 'gender_lstm_model.bin',
		gpu: false
	});

	model
		.ready()
		.then(() => {
			//console.log("Model ready");
			NProgress.done();
			$("#messageText").prop('disabled', false);
			$("#wordCnt").html("");
		})


}
function checkInput()
{
	var x = document.getElementById("messageText").value;
	if (x=="")
	{   wordsCount=0;
        $("#detectBtn").prop('disabled', true);
        //$("#messageType").html("<div class=\"alert alert-warning\">Waiting for input...</div>");
        $("#wordCnt").html("");
		
		$("#conf").html("");
		return;
	}
	
	    var re = /[0123456789`~!@#$%^&*()_|+\-=?;:'",.<>\{\}\[\]\\\/ ]/gi;
		var isSplChar = re.test(x);
		if(isSplChar)
		{
			var no_spl_char = x.replace(/[0123456789`~!@#$%^&*()_|+\-=?;:'",.<>\{\}\[\]\\\/ ]/gi, '');

		document.getElementById("messageText").value=no_spl_char;
		}
		else{
			$("#detectBtn").prop('disabled', false);
		}
		//if (x.length>2)
	     //  predictSpam();
}
function predictSpam() {
	NProgress.start();
	var x = document.getElementById("messageText").value;
	if (x=="")
		return;

	var result = TextToIndexVector(x.toLowerCase(), SEQUENCE_LENGTH);
	var seqIn = new Float32Array(result);

	model.predict({
		input: seqIn
	}).then(outputData => {
		/*
		1   0
		male female
		*/
		//console.log(outputData.output[0]);
		//console.log(outputData.output[1]);
		
		if (outputData.output[1] > outputData.output[0]) {
			
			if(outputData.output[1]>0.6)
			{$("#messageType").html("<div > <h5><img  src=\"img/man.png\"/> </h5></div>");
			var confper = Math.round( outputData.output[1]*100);
			$("#conf").html("Confidence level :"+confper+"%");
		    }
			else{
				
				var msgs=["<img  src=\"img/confused.png\"/>Hmm,I am confused!","(@_@) I dont know!"];
				var msg = msgs[Math.floor(Math.random() * msgs.length)];
				$("#messageType").html("<div ><h5>"+msg+"</h5></div>");
				//console.log(Math.abs(outputData.output[1]-outputData.output[0]));
				$("#conf").html("");
			}
	
		} else {
		
			if(outputData.output[0]>0.6)
			{
			$("#messageType").html("<div ><h5> <img  src=\"img/woman.png\"/> </h5></div>");
			var confper = Math.round(outputData.output[0]*100);
			$("#conf").html("Confidence level :"+confper+"%");
		}
		else{
			$("#messageType").html("<div ><h5><img  src=\"img/confused.png\"/>Hmm,I am confused!</h5></div>");
			//console.log(Math.abs(outputData.output[1]-outputData.output[0]));
			$("#conf").html("");
		}
			
		}
		
	
		NProgress.done();
	});

}