%% Real-time LPC-based voice pitch shifter with MIDI control.
% April 7, 2020
% Yi-Wen Liu

clear; 
close all;

fs = 16000;
framedur = 0.032;
L = fs * framedur;
Nfft = 2^nextpow2(L)*2; 
    % Note that, without zeropadding, autocorrelation
    % by IFFT will be wrong.
alpha = 0.95;
VADthres = 0.001; % voice activity detection threshold
HNRthres = 2.0; % Harmonic to noise ratio threshold. Default = 1
synthPitch = 80; % Hz
semitoneShift = 5; 
fFac = 2^(semitoneShift/12);
period = round(fs/synthPitch); % Samples, initially.
pitMin = 80;
pitMax = 500;
indMin = round(fs/pitMax);
indMax = round(fs/pitMin);

deviceReader = audioDeviceReader('SamplesPerFrame', L, 'SampleRate',fs);
deviceWriter = audioDeviceWriter('SampleRate',fs);

%% MIDI settings
maxNumNotes = 6;
availableDevices = mididevinfo;
piano = mididevice(availableDevices.input(1).ID); 
midiControlRate = 3; % once in so many frames. Default = 3.

%% Set LP parameters
P = 16; % Order of LP.
sw.deemph = 0; % option for de-emphasis. Default = 1.
sw.useDiracDelta = 0; % option for using just the Dirac delta. Default = 0.
sw.debug = 0; % turn this on if wanna show HNR and detected pitch.
sw.showWaveform = 0;

%% Initialization.
thisFrame = zeros(L,1);
thisFrameOut = zeros(P+L,1);
prevTail = zeros(P,1); % last P samples of the previous frame. For IIR filtering.
prevSigma = 0;
offset = 0;
lastSample = 0;
downramp = (L+P-1:-1:0)/L; downramp = downramp(:);
noteInfo = zeros(maxNumNotes, 3); 
    % First column = note number. If 0 -> this row is available to use.
    % Second column = velocity.
    % Third column = offset.
if sw.showWaveform
    scope = dsp.TimeScope( ...
        'SampleRate',fs, ...
        'TimeSpan',framedur, ...
        'BufferLength',fs*0.5, ...
        'YLimits',[-0.1,0.1], ...
        'TimeSpanOverrunAction',"Scroll");
end

%% Beginning of callback loop
disp('Playing can begin now... PRESS the Lowest Key (A0) to end the program.');
tic
isTerminated = 0;
frameCounter = 0;
while ~isTerminated 
    thisFrame = deviceReader(); % Read a block of input
    bigFrame = [lastSample; thisFrame];
    x_emph = bigFrame(2:end) - alpha*bigFrame(1:end-1); % emph = "emphasis"
    x_zp = [x_emph(:); zeros(Nfft-L,1)]; % zp = zeropadding
    
    %% Pitch estimation via the autocorrelation function R
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
            A = conv(A,[1 -alpha]); % convolving the denominator factors first.
        else
            [A,E,~] = levinson(R,P);
        end
        errSignal = filter(A,[1],x_emph);
            % starting the P+1'th sample, this stores the estimated glottal
            % source of the input voice.
        err_notch = errSignal(P+1:end-indMax) - errSignal(P+D+1:end-indMax+D);
        
        %% Below is a simple harmonic-to-noise ratio (HNR) estimator
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
        %% Creating an excit signal at the desired pitches and with Gaussian noise
        % Read and decypher the MIDI message
        if mod(frameCounter,midiControlRate)==0
            msgs = midireceive(piano);
            % This usually takes a longer time initially but then
            % subsequently it takes about 1-5 ms. #testingResults Apr-7-2020
            N = size(msgs,1);
            for n = 1:N
                if msgs(n).Type == 'NoteOn'
                    if msgs(n).Note == 21
                        isTerminated = 1; % The Lowest note, when hit, terminates the program.
                    end
                    jj = 1;
                    % Find the first available row and fill the information
                    while jj <= maxNumNotes
                        if noteInfo(jj,1)==0
                            noteInfo(jj,1) = msgs(n).Note;
                            noteInfo(jj,2) = msgs(n).Velocity;
                            break;
                        else
                            jj = jj+1;
                        end
                    end
                elseif msgs(n).Type == 'NoteOff'
                    jj = 1;
                    while jj <= maxNumNotes
                        if noteInfo(jj,1) == msgs(n).Note
                            noteInfo(jj,:) = 0;
                            break
                        else
                            jj = jj+1;
                        end
                    end
                end
            end
        end
        frameCounter = frameCounter + 1;
        %% cross synthesis with formant filter at MIDI pitches
        [sigma,ii] = max(abs(errSignal(P+1:P+D)));
        sigma_noise = sqrt(mean(err_notch(P+1:P+D).^2)/D); 
                % Standard deviation for the noise part.
                % division by D is an empirical scaling factor.
        gKernelProto = errSignal(P+ii:P+ii+D-1); 
                % a prototype kernel function to be tailored.
        excit = zeros(L+P,1);
                % the excitation source signal.
        tmp = noteInfo(:,1);
        idx = find(tmp); % the indices where the elements are not zero.
        if sw.debug, fprintf('# notes = %1d ',length(idx));
        end
        %% preparing the excitation signal (excit) for the harmonic parts,
        %  one note at a time.
        if HNR > HNRthres % If harmonic components are relatively strong
            if sw.debug, fprintf(', %.1f Hz ',pitch);
            end

            for whichkey = 1:length(idx)              
                notenum = noteInfo(idx(whichkey),1);
                velo = noteInfo(idx(whichkey),2);
                period = round(fs/440/2^((notenum-69)/12));
                if D > period, gKernel = gKernelProto(1:period);
                elseif D <= period
                    gKernel = [gKernelProto; zeros(period-D,1)];
                end
                offset = noteInfo(idx(whichkey),3);
                if sw.useDiracDelta
                    excit(offset+1:period:end) = excit(offset+1:period:end) + ...
                        sigma  + ...
                        (prevSigma-sigma)*downramp(offset+1:period:end);
                else
                    tmpKernel = gKernel * velo/32; % 32 is arbitrary. 
                    gKernelExt = repmat(tmpKernel,ceil((P+L)/period),1);
                    excit = excit + circshift(gKernelExt(1:P+L),offset);
                end
                noteInfo(idx(whichkey),3) = mod(offset-L,period); 
                        % To store the correct starting point for synthesis
                        % of the next frame.
            end 
        end
        %% Adding back the noise part. Assuming it is white.
        excit = excit + sigma_noise * randn(L+P,1);
        
        %% IIR filtering. 
        % Note: MATLAB's filter() function cannot be used because I could
        % not figure out a way to specify the initial conditions.
        thisFrameOut(1:P) = prevTail; % This is the initial conditions.
        for k = P+1:P+L
            slideWin = thisFrameOut(k-1:-1:k-P); % previous P samles
            slideWin = slideWin(:);
            thisFrameOut(k) = -A(2:end)*slideWin + excit(k); 
                % the classic difference equation in Signals and Systems.
        end
        lastSample = thisFrame(end);
        prevTail = thisFrameOut(L+1:L+P);
        prevSigma = sigma;

        deviceWriter(thisFrameOut(P+1:end)); % write a block of output
        if sw.showWaveform 
            scope(R(1:L)); % if wanting to view a waveform.
        end
        if sw.debug, fprintf('\n');
        end
    end
end
disp('End Signal Input')

release(deviceReader)
release(deviceWriter)
if sw.showWaveform,  release(scope);
end