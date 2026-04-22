$proxy = "http://127.0.0.1:10808"
$noProxy = "localhost,127.0.0.1,::1"

# Set both uppercase and lowercase variants because different Node/network
# stacks may read different names on Windows.
$env:HTTP_PROXY = $proxy
$env:HTTPS_PROXY = $proxy
$env:ALL_PROXY = $proxy
$env:NO_PROXY = $noProxy

$env:http_proxy = $proxy
$env:https_proxy = $proxy
$env:all_proxy = $proxy
$env:no_proxy = $noProxy

& codex @args
exit $LASTEXITCODE
