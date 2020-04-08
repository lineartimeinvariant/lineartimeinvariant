%% Real-time LPC-based monotonize-your-voice demo.
% April 3, 2020
% Yi-Wen Liu

clear; 
close all;

fs = 16000;
framedur = 0.032;
L = fs * framedur;
P = 16; % Order of LP.
Nfft = 2^nextpow2(L)*2; 
    % Note that, without zeropadding, autocorrelation
    % by IFFT will be wrong.
alpha = 0.95;
VADthres = 0.001; % voice activity detection threshold
HNRthres = 2.0; % Harmonic to noise ratio threshold. Default = 1
synthPitch = 80; % Hz
semitoneShift = 12; 
fFac = 2^(semitoneShift/12);
period = round(fs/synthPitch); % Samples, initially.
pitMin = 80;
pitMax = 500;
indMin = round(fs/pitMax);
indMax = round(fs/pitMin);
gOpenPeriod = 32; % "glottis open period". Has to be an even number.
    % empirically, this number needs to be around 32 for acceptable
    % quality. Lowering it would cause the voice to be ¨F°×
GOPwin = hann(gOpenPeriod+1); GOPwin = GOPwin(1:end-1);

%% Set LP parameters
sw.deemph = 0; % option for de-emphasis.
sw.useDiracDelta = 0; % option for using just the Dirac delta. Default = 0.
sw.useGOP = 1; % use the glottis open period to set length for the copied excitation signal.

%% other switches
sw.debug = 1;


%% Initialization.
thisFrame = zeros(L,1);
thisFrameOut = zeros(P+L,1);
prevTail = zeros(P,1);
prevSigma = 0;
offset = 0;
lastSample = 0;
downramp = (L+P-1:-1:0)/L; downramp = downramp(:);
    % First column = note number. If 0 -> this row is available to use.
    % Second column = offset.
last3per = [80 80 80]; % storing the last 3 frame's period.

deviceReader = audioDeviceReader('SamplesPerFrame', L, 'SampleRate',fs,...
    'BitDepth','16-bit integer');
deviceWriter = audioDeviceWriter('SampleRate',fs,...
    'BitDepth','16-bit integer');


scope = dsp.TimeScope( ...                     
    'SampleRate',fs, ...       
    'TimeSpan',framedur, ...                             
    'BufferLength',fs*0.5,...
    'YLimits',[-0.01,0.01]) %, ...                         
    %'TimeSpanOverrunAction',"Scroll");            

disp('Begin Signal Input... Press Ctrl-C to end anytime.')
tic
while 1 % or while toc < N second
    thisFrame = deviceReader();
    bigFrame = [lastSample; thisFrame];
    x_emph = bigFrame(2:end) - alpha*bigFrame(1:end-1);
    x_zp = [x_emph(:); zeros(Nfft-L,1)];
    
    if max(abs(x_emph)) > VADthres
        X = fft(x_zp);
        R = ifft(X.*conj(X));
        [~,findex] = max(R(end-indMin:-1:end-indMax));
        D = indMin + findex - 1; % Delay of notch filter
        pitch = fs/D;
        % This pitch detection works fine except
        % (1) Pitch is discretized by fs/ integers
        % (2) Often got octave confusion for the vowel /u/
        % (3) Come to think about it, probably a longer window (>=32 ms) is
        %     preferrable.
        %%
        if sw.deemph
            [A,E,~] = levinson(R,P-1); 
            %[A,E] = lpc(x_emph,P-1); % Take the second half for LP analysis.
            A = conv(A,[1 -alpha]);
        else
            [A,E,~] = levinson(R,P);
        end
        errSignal = filter(A,[1],x_emph);
        err_notch = errSignal(P+1:end-indMax) - errSignal(P+D+1:end-indMax+D);
        
        %% Below is a simple harmonic-to-noise ratio estimation
        % It requires L to be at least twice longer than indMin.
        ASinglePeriod = zeros(D,1);
        noise = zeros(D,1);
        numPeriods = floor(L/D);
        if mod(numPeriods,2) == 1
            numPeriods = numPeriods -1;
        end
        tmp = 0;
        for cc = 1:numPeriods
            ASinglePeriod = ASinglePeriod + x_emph(tmp+1:tmp+D);
            noise = noise + (-1)^cc * x_emph(tmp+1:tmp+D);
            tmp = tmp + D;
        end
        HNR = sum(ASinglePeriod.^2)/sum(noise.^2) - 1;
        if sw.debug, fprintf('HNR = %.3f ', HNR);
        end
        %% Creating an excit signal with impulses and Gaussian noise
        if sw.useDiracDelta
            %sigma = sqrt(mean(errSignal(P+1:P+D).^2));
            %sigma = sigma*sqrt(D); % sqrt(D) is arbitrary.
            sigma = max(abs(errSignal(P+1:end)));
        else
            [~,ii] = max(abs(errSignal(P+1:P+D)));
        end
        sigma_noise = sqrt(mean(err_notch(P+1:end).^2)/numPeriods);
        %sigma_noise = sigma_noise/5; % arbitrary scaling
        excit = zeros(L+P,1);
        if HNR > HNRthres
            if sw.debug, fprintf('Pitch = %.1f Hz ',pitch);
            end
            %period = round(D/fFac);
            last3per = circshift(last3per,1);
            last3per(1) = D;
            period = round(median(last3per)/fFac); 
                % median filtering to remove single-frame octave confusion.
            
            if sw.useDiracDelta
                excit(offset+1:period:end) = sigma + ...
                    (prevSigma-sigma)*downramp(offset+1:period:end);
                prevSigma = sigma;

            else
                if sw.useGOP
                    gKernel = zeros(period,1);
                    gKernel(1:gOpenPeriod) = GOPwin.*...
                        errSignal(P+D+ii-gOpenPeriod/2: P+D+ii+gOpenPeriod/2-1);
                else
                    gKernel = errSignal(P+ii:P+ii+D-1);
                    if D >= period, gKernel = gKernel(1:period);
                    else
                        gKernel = [gKernel; zeros(period-D,1)];
                    end
                end
                gKernelExt = repmat(gKernel,ceil((P+L)/period),1);
                excit = circshift(gKernelExt(1:P+L),offset);
                % it seems downramp causes some problem. Support suspended.
                % 4/6/2020.
            end
        end
        excit = excit + sigma_noise * randn(L+P,1);
        if sw.debug, fprintf('\n');
        end
        thisFrameOut(1:P) = prevTail;
        for k = P+1:P+L
            slideWin = thisFrameOut(k-1:-1:k-P);
            slideWin = slideWin(:);
            thisFrameOut(k) = -A(2:end)*slideWin + excit(k);
        end
        lastSample = thisFrame(end);
        prevTail = thisFrameOut(L+1:L+P);

        offset = mod(offset - L, period);
        deviceWriter(thisFrameOut(P+1:end)); % starting the p+1
        %scope(errSignal(P+1:end));
        %scope(R(1:L));
        
    end
end
disp('End Signal Input')

release(deviceReader)
release(deviceWriter)
release(scope)