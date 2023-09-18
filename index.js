'use strict';
/** @type {HTMLSpanElement} */
let mspfCounter = document.getElementById('mspf-counter');
/** @type {HTMLSpanElement} */
let fpsCounter = document.getElementById('fps-counter');

/** @type {HTMLCanvasElement} */
const canvas = document.getElementById("glcanv");
/** @type {WebGL2RenderingContext} */
let gl = canvas.getContext('webgl2', {
    antialias: true
}); 

const vertexSource = `#version 300 es

in vec2 a_position;
in vec2 a_texCoord;

out vec2 texCoord;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    texCoord = a_texCoord;
}
`;


const fragmentSource = `#version 300 es
#define PI 3.141592
#ifdef GL_ES
precision highp float;
#endif

// replacement for gl_FragColor
out vec4 fragColor;
in highp vec2 texCoord;
uniform float time;
uniform uvec2 viewportSize; 
// uniform vec4 userParam;
// uniform sampler2D userTexture;
// uniform sampler2D lightTexture;


//
// Description : Array and textureless GLSL 2D/3D/4D simplex 
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20201014 (stegu)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
// 

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+10.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v)
  { 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 105.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

void main() {
    float paramA = 6.f;
    float paramB = 1.f;
    float paramC = 4.f;
    float tmfd = float(time) / 16000.0;
    vec2 goodSize = vec2(viewportSize); 
    vec2 actc = texCoord * (goodSize / min(goodSize.x, goodSize.y));
    vec3 colorA = vec3(0.3f, 0.1f, 0.8f);
    vec3 colorB = vec3(0.5f, 0.2f, 1.0f);
    //fragColor = vec4(actc, 0.0, 1.0);
    float fac1 = snoise(vec3(actc * vec2(paramA), tmfd));
    float fac2 = round(snoise(vec3((actc * vec2(paramB)) + vec2(paramC * fac1), tmfd)) * 2.0) / 2.0;
    fragColor = vec4(vec3(mix(colorA, colorB, fac2)), 1.0);
}
`

/** @type {WebGLShader} */
let vertexShader = null;
/** @type {WebGLProgram} */
let shaderProgram = null;
/** @type {WebGLBuffer} */
let positionBuffer = null;
/** @type {WebGLBuffer} */
let texCoordBuffer = null;
/** @type {WebGLBuffer} */
let indexBuffer = null;

const attributeLoc = {
    /** @property {GLuint>} */
    position: null,
    /** @property {GLuint>} */
    texCoord: null
}

const uniform = {
    /** @property {GLuint>} */
    viewportSize: null,
    /** @property {GLuint>} */
    userParam: null,
    // cursor: null,
    /** @property {GLuint>} */
    time: null,
};

/** @type {GLuint} */
let userTexture = null;

let compileTime = Date.now();

const vertices = new Float32Array([
    -1.0, 1.0,  // top left
    1.0, 1.0,   // top right
    -1.0, -1.0, // bottom left
    1.0, -1.0   // bottom right
]);

const coords = new Float32Array([
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0
]);

const indices = new Uint16Array([
    0, 1, 2,
    1, 2, 3
]);

requestAnimationFrame = requestAnimationFrame || webkitRequestAnimationFrame;

function updateShaders() {
    /** @type {WebGLProgram} */
    let p = gl.createProgram();
    /** @type {WebGLShader} */
    let fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fragmentSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
        let e = new Error(gl.getShaderInfoLog(fs));
        gl.deleteShader(fs);
        gl.deleteProgram(p);
        throw e;
    };
    gl.attachShader(p, vertexShader);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        let e = new Error(gl.getProgramInfoLog(p));
        gl.detachShader(p, fs);
        gl.detachShader(p, vertexShader);
        gl.deleteShader(fs);
        gl.deleteProgram(p);
        throw e;
    };
    gl.detachShader(p, fs);
    gl.deleteShader(fs);
    gl.deleteProgram(shaderProgram);
    shaderProgram = p;
    // attributes
    attributeLoc.position = gl.getAttribLocation(shaderProgram, 'a_position');
    attributeLoc.texCoord = gl.getAttribLocation(shaderProgram, 'a_texCoord');
    // uniforms
    // uniform.userParam = gl.getUniformLocation(shaderProgram, 'userParam');
    uniform.viewportSize = gl.getUniformLocation(shaderProgram, 'viewportSize');
    uniform.time = gl.getUniformLocation(shaderProgram, 'time');
    compileTime = Date.now() + (Math.random() * 10000);
}

function main() {
    vertexShader = gl.createShader(gl.VERTEX_SHADER);

    gl.shaderSource(vertexShader, vertexSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
        let e = new Error(gl.getShaderInfoLog(vertexShader));
        gl.deleteShader(vertexShader);
        gl.deleteProgram(shaderProgram);
        throw e;
    };

    updateShaders();

    positionBuffer = gl.createBuffer();
    texCoordBuffer = gl.createBuffer();
    indexBuffer = gl.createBuffer();

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, coords, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(attributeLoc.position);
    gl.vertexAttribPointer(
        attributeLoc.position,
        2, // 2 values per vertex shader iteration
        gl.FLOAT, // data is 32bit floats
        false,        // don't normalize
        0,            // stride (0 = auto)
        0,            // offset into buffer
    );

    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.enableVertexAttribArray(attributeLoc.texCoord);
    gl.vertexAttribPointer(
        attributeLoc.texCoord,
        2,
        gl.FLOAT,
        false, // normalize
        0,     // stride (0 = auto)
        0,     // offset into buffer
    );

    // userTexture = gl.createTexture();
    // gl.bindTexture(gl.TEXTURE_2D, userTexture);
    // const level = 0;
    // const internalFormat = gl.RGBA;
    // const border = 0;
    // const srcFormat = gl.RGBA;
    // const srcType = gl.UNSIGNED_BYTE;
    // // const pixel = new Uint8Array([0, 0, 255, 255]); // opaque blue
    // gl.texImage2D(
    //     gl.TEXTURE_2D,
    //     level,
    //     internalFormat,
    //     imge.width,
    //     imge.height,
    //     border,
    //     srcFormat,
    //     srcType,
    //     imge
    // );
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    requestAnimationFrame(tick);
}

let lastTickTime = Date.now();
let lastUpdateTime = 0;

// let tickTiming = {
//     fps: 0,
//     mspf: 0
// };

function tick() {
    let mspf = Date.now() - lastTickTime;
    let fps = Math.floor(1000 / mspf);
    lastTickTime = Date.now();
    // tickTiming.fps += fps;
    // tickTiming.mspf += mspf;

    console.log("frame time: ", mspf, `ms (approximately ${fps}fps)`);

    if (lastUpdateTime + 500 <= Date.now()) {
        mspfCounter.textContent = mspf;
        fpsCounter.textContent = fps;
        lastUpdateTime = Date.now();
    }

    // gl.clearColor(0.125, 0.25, 0.75, 1);

    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(shaderProgram);

    // gl.bindTexture(gl.TEXTURE_2D, userTexture);
    // gl.activeTexture(gl.TEXTURE0);
    // gl.bindTexture(gl.TEXTURE_2D, lightTexture);
    // gl.activeTexture(gl.TEXTURE1);
    // gl.uniform4f(uniform.userParam, customX, customY, customZ, customW);
    gl.uniform2ui(uniform.viewportSize, gl.canvas.width, gl.canvas.height);
    gl.uniform1f(uniform.time, Date.now() - compileTime);

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    gl.drawElements(
        gl.TRIANGLES,
        6,                 // num vertices to process
        gl.UNSIGNED_SHORT, // type of indices
        0,                 // offset on bytes to indices
    );
    
    gl.finish();

    requestAnimationFrame(tick);
}

function handleResize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

window.addEventListener('resize', handleResize);
handleResize();
window.addEventListener('load', main);
// main();
